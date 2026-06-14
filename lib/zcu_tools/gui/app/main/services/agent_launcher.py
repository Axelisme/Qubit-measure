"""Qt-free helpers to launch an interactive ``claude`` in the system terminal.

The final embedded-agent design (Round 1) is: one GUI button opens the OS's
default terminal emulator running the real interactive ``claude`` CLI, pointed
at a loopback measure-gui MCP server (which auto-attaches to the already-running
GUI via session discovery). There is no in-process transcript, no PTY, and no
stream-json bridge — the user talks to claude directly in their own terminal.

This module is Qt-free and holds no process state: it builds the argv / MCP
config / Python launcher and spawns a detached terminal via ``subprocess.Popen``.

The MCP loopback config and the embedded system prompt own the "GUI already
attached" contract (the live ``mcp__measure-gui__*`` tools auto-attach to the
running GUI, so the agent must not issue its own connect). The launcher
optionally appends a live GUI-state snapshot (from the Controller) to the
embedded prompt so the agent starts already knowing the current project /
context / SoC / open tabs. The legacy in-process stream-json runner is gone
(Round 2); this is the only embedded-agent path.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Persisted session list. Replaces the old single-id ``agent_last_session`` file.
# Stores a JSON array of {"session_id": str, "created": float} records (most
# recently launched first). Capped at _SESSION_CAP entries.
_SESSIONS_FILE = Path.home() / ".cache" / "zcu-tools" / "agent_sessions.json"
_SESSION_CAP = 30

# Maximum label length extracted from the first user message in a jsonl.
_LABEL_MAX_CHARS = 60

# Appended to claude's system prompt. The mcp__measure-gui__* tools auto-attach
# to the live GUI (lazy auto-connect in the MCP server), so the agent must NOT
# issue any connect itself — doing so conflicts with the user-owned SoC link.
# Kept short for token economy / prompt-cache friendliness.
_EMBEDDED_SYSTEM_PROMPT = (
    "You are operating a measure-gui that is already running. The "
    "mcp__measure-gui__* tools are already attached to it — do NOT call "
    "gui_connect, gui_connect_start, or any other connect; the SoC link is the "
    "user's decision, not yours. To see the current state call gui_state_check / "
    "gui_soc_info and other query tools. Do NOT use ToolSearch or the "
    "run-measure-gui Skill — the GUI tools are already available. Be concise and "
    "act directly; do not narrate every step."
)


# ---------------------------------------------------------------------------
# ResumableSession
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumableSession:
    """A previously launched agent session that can be resumed.

    ``last_active`` is the Unix epoch float of the jsonl mtime (or the
    ``created`` timestamp from the store when the jsonl is absent).
    ``label`` is a best-effort excerpt of the first user message (≤60 chars),
    falling back to the first 8 chars of the session id when the jsonl is
    unreadable.
    """

    session_id: str
    last_active: float
    label: str


# ---------------------------------------------------------------------------
# Session store helpers
# ---------------------------------------------------------------------------


def _read_sessions_store() -> list[dict[str, object]]:
    """Return the raw list from ``_SESSIONS_FILE``, or [] on any read/parse error."""
    try:
        raw = _SESSIONS_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        return data  # type: ignore[return-value]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _write_sessions_store(records: list[dict[str, object]]) -> None:
    """Atomically write ``records`` to ``_SESSIONS_FILE``."""
    _SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SESSIONS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    os.replace(tmp, _SESSIONS_FILE)


def record_launched_session(session_id: str) -> None:
    """Append ``session_id`` to the persistent session list (dedup, cap at 30).

    Most-recently-launched is kept at index 0. An existing entry with the same
    id is moved to the front rather than duplicated. The list is capped at
    ``_SESSION_CAP`` entries so the file does not grow unboundedly.
    """
    if not session_id:
        raise ValueError("session_id must be non-empty")
    records = _read_sessions_store()
    # Remove any existing entry with the same id (dedup).
    records = [r for r in records if r.get("session_id") != session_id]
    # Prepend the new entry so the list is sorted newest-first.
    records.insert(0, {"session_id": session_id, "created": time.time()})
    # Cap to avoid unbounded growth.
    records = records[:_SESSION_CAP]
    _write_sessions_store(records)


# ---------------------------------------------------------------------------
# claude project directory
# ---------------------------------------------------------------------------


def claude_project_dir(repo_root: str) -> Path:
    """Return the ``~/.claude/projects/<slug>`` directory for ``repo_root``.

    The slug encoding matches claude's own convention on POSIX: take the absolute
    path of ``repo_root`` and replace every character outside ``[A-Za-z0-9-]``
    with ``-``. For example ``/home/user/my.repo`` → ``-home-user-my-repo``.

    NOTE (Windows-verify): claude's actual slug encoding for Windows paths (drive
    letter ``C:``, backslash separators, the colon) has not been verified against
    a real claude install. If it diverges, the ``<id>.jsonl`` lookup below will
    miss and ``list_resumable_sessions`` degrades gracefully — it still returns
    every recorded session using the stored ``created`` timestamp and an
    ``<id[:8]>`` label (no crash, just no extracted first-message label / mtime).
    """
    abspath = os.path.abspath(repo_root)
    slug = re.sub(r"[^A-Za-z0-9-]", "-", abspath)
    return Path.home() / ".claude" / "projects" / slug


# ---------------------------------------------------------------------------
# Label extraction from a claude jsonl
# ---------------------------------------------------------------------------


def _extract_label_from_jsonl(jsonl_path: Path) -> str | None:
    """Return the first user message text from a claude session jsonl, or None.

    Scans lines until it finds a ``{"type": "user", ...}`` entry whose
    ``message.content`` yields a non-empty text string. The result is truncated
    to ``_LABEL_MAX_CHARS`` characters. Any malformed line is silently skipped
    (never raises).

    ``content`` may be a plain string or a list of content blocks
    (``[{"type": "text", "text": "..."}]``).
    """
    try:
        with jsonl_path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") != "user":
                    continue
                message = obj.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                text: str | None = None
                if isinstance(content, str):
                    text = content.strip() or None
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            candidate = block.get("text", "")
                            if isinstance(candidate, str) and candidate.strip():
                                text = candidate.strip()
                                break
                if text:
                    return text[:_LABEL_MAX_CHARS]
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# list_resumable_sessions
# ---------------------------------------------------------------------------


def list_resumable_sessions(repo_root: str) -> list[ResumableSession]:
    """Return previously launched sessions that can be resumed, newest first.

    Only sessions we have launched ourselves (recorded in ``_SESSIONS_FILE``)
    are returned — the claude project directory typically contains many unrelated
    dev sessions. For each recorded id:

    - If the corresponding ``<id>.jsonl`` exists: ``last_active`` = mtime,
      ``label`` = first user message (truncated) or ``<id[:8]>`` fallback.
    - If the jsonl is absent: ``last_active`` = stored ``created`` timestamp,
      ``label`` = ``<id[:8]>``.

    Returns ``[]`` when the store is empty or the project directory is absent.
    """
    records = _read_sessions_store()
    if not records:
        return []

    project_dir = claude_project_dir(repo_root)
    sessions: list[ResumableSession] = []

    for record in records:
        session_id = record.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            continue
        created: float = float(record.get("created", 0.0))  # type: ignore[arg-type]

        jsonl = project_dir / f"{session_id}.jsonl"
        if jsonl.exists():
            try:
                last_active = jsonl.stat().st_mtime
            except OSError:
                last_active = created
            label = _extract_label_from_jsonl(jsonl) or session_id[:8]
        else:
            last_active = created
            label = session_id[:8]

        sessions.append(
            ResumableSession(
                session_id=session_id,
                last_active=last_active,
                label=label,
            )
        )

    # Sort newest-first by last_active.
    sessions.sort(key=lambda s: s.last_active, reverse=True)
    return sessions


# ---------------------------------------------------------------------------
# Core argv / config builders
# ---------------------------------------------------------------------------


def build_loopback_mcp_config(repo_root: str) -> str:
    """Write a temp ``.mcp.json`` that points claude at the measure-gui MCP server.

    The server entry mirrors the repo's top-level ``.mcp.json``: it uses
    ``uv run --extra gui python lib/zcu_tools/mcp/measure/server.py`` with the
    repo root as cwd. When the spawned MCP server starts, it reads the session
    discovery file (``~/.cache/zcu-tools/sessions/measure.json``) to locate the
    already-running GUI's TCP port.

    Returns the absolute path to the written config file.
    """
    config: dict[str, object] = {
        "mcpServers": {
            "measure-gui": {
                "type": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "--extra",
                    "gui",
                    "python",
                    "lib/zcu_tools/mcp/measure/server.py",
                ],
                # cwd must be the repo root so uv finds pyproject.toml and the
                # relative script path resolves correctly.
                "cwd": repo_root,
                "env": {},
            }
        }
    }
    path = Path(tempfile.gettempdir()) / "zcu_agent_mcp.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(path)


def new_session_id() -> str:
    """Return a fresh dashed UUID for claude's ``--session-id``.

    ``claude --session-id`` requires a *valid UUID* (verified via ``claude
    --help``: ``--session-id <uuid>``), so this returns the canonical dashed
    form, not a bare hex digest.
    """
    return str(uuid.uuid4())


def _find_desktop_bundled_claude() -> str | None:
    """Return the newest Claude Desktop-bundled ``claude.exe`` path, or None.

    Claude Desktop ships the Claude Code CLI under
    ``%APPDATA%\\Claude\\claude-code\\<version>\\claude.exe`` — a version-pinned
    directory that is *not* on PATH, so ``shutil.which("claude")`` cannot find it
    on a Desktop-only install. This returns the highest-version ``claude.exe``
    that exists, or None when Desktop is absent / the layout differs.

    Windows-only by construction (relies on ``%APPDATA%``); callers gate on
    ``sys.platform == "win32"``.
    """
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    base = Path(appdata) / "Claude" / "claude-code"
    if not base.is_dir():
        return None
    candidates: list[tuple[tuple[int, ...], str]] = []
    for child in base.iterdir():
        exe = child / "claude.exe"
        if not exe.is_file():
            continue
        # Order by the dir name parsed as a dotted version so 2.1.170 wins over
        # 2.1.9 (lexicographic order would get that backwards). Parsing stops at
        # the first non-numeric part, so a well-formed version sorts highest.
        version: list[int] = []
        for part in child.name.split("."):
            if not part.isdigit():
                break
            version.append(int(part))
        candidates.append((tuple(version), str(exe)))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[-1][1]


def resolve_agent_command() -> str:
    """Resolve argv[0] for the agent CLI — the binary the launcher execs.

    Precedence:
      1. ``ZCU_AGENT_CMD`` (explicit override, any platform) — e.g. swap in codex.
      2. On Windows: Claude Desktop's bundled CLI (newest
         ``%APPDATA%\\Claude\\claude-code\\*\\claude.exe``). Desktop installs do
         not put ``claude`` on PATH, so this is preferred before the PATH lookup.
      3. The bare ``"claude"`` — resolved via PATH in the launcher (a standalone
         Claude Code CLI install). The launcher Fast-Fails if it is absent.

    The Windows order is therefore: Desktop-bundled CLI → PATH ``claude`` →
    Fast-Fail; other platforms: ``ZCU_AGENT_CMD`` → PATH ``claude`` → Fast-Fail.
    """
    override = os.environ.get("ZCU_AGENT_CMD")
    if override:
        return override
    if sys.platform == "win32":
        bundled = _find_desktop_bundled_claude()
        if bundled is not None:
            return bundled
    return "claude"


def build_claude_argv(
    mcp_config_path: str,
    *,
    resume_session_id: str | None = None,
    new_session_id: str | None = None,
    allowed_tools: str = "mcp__measure-gui__*",
    state_context: str | None = None,
) -> list[str]:
    """Build argv for an *interactive* ``claude`` child (no ``-p`` print mode).

    The child runs in its normal TTY mode (so the user sees the real terminal
    UI) with:
      - ``--mcp-config <path>`` loopback to the running GUI's MCP server,
      - ``--allowedTools <allowed_tools>`` to restrict tool access,
      - ``--append-system-prompt`` carrying the embedded "GUI already attached"
        instruction, plus the live GUI-state snapshot when ``state_context`` is
        given (appended after the embedded prompt so the agent starts already
        knowing the project / context / SoC / open tabs — Round 2).

    Session continuity: ``resume_session_id`` (if given) wins and maps to
    ``--resume <id>``; otherwise ``new_session_id`` (if given) maps to
    ``--session-id <id>``. Both ``None`` is valid — claude manages the session.

    argv[0] is ``resolve_agent_command()`` (``ZCU_AGENT_CMD`` override → on
    Windows the Claude Desktop-bundled CLI → bare ``claude`` resolved via PATH).
    """
    system_prompt = _EMBEDDED_SYSTEM_PROMPT
    if state_context is not None:
        system_prompt = _EMBEDDED_SYSTEM_PROMPT + "\n\n" + state_context
    argv = [
        resolve_agent_command(),
        "--mcp-config",
        mcp_config_path,
        "--allowedTools",
        allowed_tools,
        "--append-system-prompt",
        system_prompt,
    ]
    if resume_session_id is not None:
        argv += ["--resume", resume_session_id]
    elif new_session_id is not None:
        argv += ["--session-id", new_session_id]
    return argv


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def build_python_launcher_source(repo_root: str, argv: list[str]) -> str:
    """Return the source of a cross-platform Python launcher that execs ``argv``.

    The launcher embeds ``argv`` and ``repo_root`` as *data* via ``json.dumps``
    (whose output is a valid Python literal), so the multi-line
    ``--append-system-prompt`` value — newlines, quotes, anything — survives
    without any shell/cmd/bat quoting. This is the whole reason for the indirection:
    a ``.bat`` could not carry a multi-line GUI-state snapshot without breaking
    cmd syntax, and ``.sh`` quoting was POSIX-only.

    The generated launcher chdirs into the repo, drops ``ANTHROPIC_API_KEY`` so
    claude uses subscription auth, resolves the binary via ``shutil.which`` (on
    Windows ``claude`` is usually ``claude.cmd`` — ``os.execv`` alone would not
    find it on PATH), and Fast-Fails with a clear message when the binary is
    absent. It is run as ``<python> <launcher.py>``, so it needs no execute bit.
    """
    argv_literal = json.dumps(argv)
    cwd_literal = json.dumps(os.path.abspath(repo_root))
    # ARGV / CWD are written as json.dumps output, i.e. valid Python list/str
    # literals — this is what makes the multi-line prompt safe.
    return (
        "import os, shutil, sys\n"
        f"ARGV = {argv_literal}\n"
        f"CWD = {cwd_literal}\n"
        "os.chdir(CWD)\n"
        "os.environ.pop('ANTHROPIC_API_KEY', None)\n"
        "_bin = shutil.which(ARGV[0])\n"
        "if _bin is None:\n"
        "    sys.exit(\n"
        '        f"zcu agent launcher: command not found on PATH: {ARGV[0]!r}. "\n'
        '        "Install it or set ZCU_AGENT_CMD to the right CLI."\n'
        "    )\n"
        # execv replaces this process with the resolved binary, preserving the
        # terminal's TTY so claude runs in its normal interactive UI.
        "os.execv(_bin, ARGV)\n"
    )


def _write_python_launcher(repo_root: str, argv: list[str]) -> str:
    """Write the cross-platform Python launcher to a temp ``.py``; return its path.

    A fresh per-launch filename avoids clobbering a concurrently launched agent.
    """
    source = build_python_launcher_source(repo_root, argv)
    fd, path = tempfile.mkstemp(prefix="zcu_agent_launch_", suffix=".py")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(source)
    return path


def _spawn_terminal_argv(launcher_path: str) -> list[str]:
    """Resolve the OS terminal-emulator argv that runs ``launcher_path``.

    The command run inside the terminal is always the two tokens
    ``[sys.executable, launcher_path]`` — all argv complexity lives inside the
    launcher, so the terminal command stays trivial to quote.

    ``ZCU_AGENT_TERMINAL`` overrides the terminal on both Windows and Linux. macOS
    uses Terminal.app; Windows runs the launcher directly (the caller attaches a
    fresh window via CREATE_NEW_CONSOLE) rather than the Store-packaged Windows
    Terminal, whose AppData\\Roaming sandbox hides the Desktop-bundled CLI (see the
    win32 branch). Fast-fails with a RuntimeError teaching the user to set
    ``ZCU_AGENT_TERMINAL`` when no known terminal is found on Linux.
    """
    py = sys.executable

    if sys.platform == "darwin":
        # macOS-verify: Terminal.app's ``do script`` takes a *shell string*, so
        # the python + launcher paths are double-quoted into it. AppleScript
        # string literals escape embedded double quotes with a backslash.
        shell_cmd = f"{_applescript_quote(py)} {_applescript_quote(launcher_path)}"
        script = f'tell application "Terminal" to do script "{shell_cmd}"'
        return ["osascript", "-e", script]

    if sys.platform == "win32":
        # An explicit terminal override wins (mirrors the Linux branch): a power
        # user with a non-sandboxed terminal can force it.
        override = os.environ.get("ZCU_AGENT_TERMINAL")
        if override:
            return [override, py, launcher_path]
        # Default: run the launcher directly and let the caller attach a fresh
        # console via CREATE_NEW_CONSOLE (see launch_agent_terminal). This beats
        # both ``wt`` and ``cmd /c start``:
        #   - ``wt`` is a UWP app and virtualizes AppData\Roaming for the children
        #     it spawns, so a wt-launched agent cannot read the Claude Desktop CLI
        #     under %APPDATA%\Roaming\Claude (os.path.exists is False → Fast-Fail).
        #   - ``cmd /c start "" "<py>" "<launcher>"`` double-quotes when subprocess
        #     re-escapes the already-quoted tokens, corrupting the path.
        # A direct Popen from this normal (non-packaged) process has no AppData
        # sandbox and lets subprocess quote the two real paths correctly. See
        # ADR-0024 (Windows terminal sandbox).
        return [py, launcher_path]

    # Linux / other POSIX. An explicit override wins so headless/custom setups
    # can point at any terminal: ZCU_AGENT_TERMINAL is the full program name and
    # the python + launcher are appended as the trailing args.
    override = os.environ.get("ZCU_AGENT_TERMINAL")
    if override:
        return [override, py, launcher_path]
    # gnome-terminal/konsole/xterm all run a program with its args; the flag form
    # differs per terminal. Passed as a list to Popen, so no shell quoting needed.
    if shutil.which("gnome-terminal") is not None:
        return ["gnome-terminal", "--", py, launcher_path]
    if shutil.which("konsole") is not None:
        return ["konsole", "-e", py, launcher_path]
    if shutil.which("xterm") is not None:
        return ["xterm", "-e", py, launcher_path]
    if shutil.which("x-terminal-emulator") is not None:
        return ["x-terminal-emulator", "-e", py, launcher_path]
    raise RuntimeError(
        "No terminal emulator found (looked for gnome-terminal, konsole, xterm, "
        "x-terminal-emulator). Set the ZCU_AGENT_TERMINAL environment variable to "
        "your terminal program (it is invoked as "
        "'<terminal> <python> <launcher.py>')."
    )


def _applescript_quote(value: str) -> str:
    """Escape ``value`` for embedding inside an AppleScript double-quoted string.

    AppleScript string literals escape ``\\`` and ``"`` with a backslash. The
    result is wrapped in double quotes so a space-containing path stays one token
    inside Terminal.app's ``do script`` shell string.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'\\"{escaped}\\"'


# ---------------------------------------------------------------------------
# Main launch entry point
# ---------------------------------------------------------------------------


def launch_agent_terminal(
    repo_root: str,
    *,
    resume_session_id: str | None = None,
    state_context: str | None = None,
) -> str:
    """Open the system terminal running interactive ``claude`` against the GUI.

    When ``resume_session_id`` is given the terminal resumes that specific
    session (``--resume <id>``). Otherwise a fresh UUID is generated, recorded
    in the persistent session list, and passed via ``--session-id <id>``.

    Returns the session id that was used (the given resume id or the new id).

    ``state_context`` (a live GUI-state snapshot from the Controller) is appended
    to the embedded system prompt so the agent starts already knowing the current
    project / context / SoC / open tabs. ``None`` keeps only the static prompt.

    The spawn is detached (``subprocess.Popen``, not waited). The terminal runs a
    cross-platform Python launcher (``<python> <launcher.py>``) that chdirs into
    the repo, drops ``ANTHROPIC_API_KEY`` so claude uses subscription auth, and
    execs claude; the env passed to ``Popen`` also drops the key (double safety,
    so even the terminal process never sees it). Cross-platform branches Fast-Fail
    rather than silently degrading.
    """
    mcp_config_path = build_loopback_mcp_config(repo_root)

    if resume_session_id is not None:
        session_id = resume_session_id
        argv = build_claude_argv(
            mcp_config_path,
            resume_session_id=session_id,
            state_context=state_context,
        )
        # Resuming an existing session: do not re-record it (it is already in
        # the store from when it was first launched).
    else:
        session_id = new_session_id()
        record_launched_session(session_id)
        argv = build_claude_argv(
            mcp_config_path,
            new_session_id=session_id,
            state_context=state_context,
        )

    launcher_path = _write_python_launcher(repo_root, argv)
    spawn_argv = _spawn_terminal_argv(launcher_path)

    env = os.environ.copy()
    # Subscription auth: claude must NOT see an API key (mirrors agent_runner).
    env.pop("ANTHROPIC_API_KEY", None)

    # On Windows, the default spawn is the launcher itself ([py, launcher]); give
    # it its own console window via CREATE_NEW_CONSOLE so claude's TUI has a real
    # terminal. Skipped when ZCU_AGENT_TERMINAL is set (that terminal owns its own
    # window) and a no-op (0) on non-Windows / when the flag is unavailable.
    creationflags = 0
    if sys.platform == "win32" and not os.environ.get("ZCU_AGENT_TERMINAL"):
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

    # detached: do not wait
    subprocess.Popen(spawn_argv, env=env, creationflags=creationflags)
    return session_id
