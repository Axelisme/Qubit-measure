"""Qt-free helpers to launch an interactive ``claude`` in the system terminal.

The final embedded-agent design (Round 1) is: one GUI button opens the OS's
default terminal emulator running the real interactive ``claude`` CLI, pointed
at a loopback measure-gui MCP server (which auto-attaches to the already-running
GUI via session discovery). There is no in-process transcript, no PTY, and no
stream-json bridge — the user talks to claude directly in their own terminal.

This module is Qt-free and holds no process state: it builds the argv / MCP
config / launch script and spawns a detached terminal via ``subprocess.Popen``.

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
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# argv[0] for the agent CLI. ``claude`` by default; ``ZCU_AGENT_CMD`` overrides
# it so a future codex-style CLI can be swapped in without code changes.
AGENT_CMD: str = os.environ.get("ZCU_AGENT_CMD", "claude")

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

    The slug encoding matches claude's own convention: take the absolute path of
    ``repo_root`` and replace every character outside ``[A-Za-z0-9-]`` with
    ``-``. For example ``/home/user/my.repo`` → ``-home-user-my-repo``.
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

    argv[0] is ``AGENT_CMD`` (``claude`` by default, ``ZCU_AGENT_CMD`` override).
    """
    system_prompt = _EMBEDDED_SYSTEM_PROMPT
    if state_context is not None:
        system_prompt = _EMBEDDED_SYSTEM_PROMPT + "\n\n" + state_context
    argv = [
        AGENT_CMD,
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


def _write_launch_script(repo_root: str, argv: list[str]) -> str:
    """Write a temp launcher script that ``cd``s into the repo and execs argv.

    Spawning a terminal that runs a *script* (rather than passing the full argv
    through the terminal's own ``-e`` arg) sidesteps per-terminal quoting rules
    for the long ``--append-system-prompt`` value. The script is shell-quoted
    so embedded spaces/quotes survive.

    POSIX: a ``.sh`` running ``cd <repo> && exec <argv...>``.
    Windows: a ``.bat`` running ``cd /d <repo>`` then the argv.
    """
    tmp_dir = Path(tempfile.gettempdir())
    if sys.platform == "win32":
        # cmd.exe quoting: wrap each arg in double quotes; embedded quotes are
        # doubled. argv values here contain no double quotes, so this is safe.
        quoted = " ".join('"' + a.replace('"', '""') + '"' for a in argv)
        script = tmp_dir / "zcu_agent_launch.bat"
        body = f'@echo off\r\ncd /d "{repo_root}"\r\n{quoted}\r\n'
        script.write_text(body, encoding="utf-8")
        return str(script)
    quoted = " ".join(shlex.quote(a) for a in argv)
    script = tmp_dir / "zcu_agent_launch.sh"
    body = f"#!/bin/sh\ncd {shlex.quote(repo_root)} && exec {quoted}\n"
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def _spawn_terminal_argv(script_path: str) -> list[str]:
    """Resolve the OS terminal-emulator argv that runs ``script_path``.

    Fast-fails with a RuntimeError teaching the user to set ``ZCU_AGENT_TERMINAL``
    when no known terminal is found (Linux). macOS/Windows branches use their
    platform-standard launchers.
    """
    if sys.platform == "darwin":
        # ``open -a Terminal <script>`` opens the script in Terminal.app.
        return ["open", "-a", "Terminal", script_path]

    if sys.platform == "win32":
        wt = shutil.which("wt")
        if wt is not None:
            return [wt, "cmd", "/k", script_path]
        # ``start "" cmd /k <bat>`` opens a new console window running the bat.
        return ["cmd", "/c", "start", "", "cmd", "/k", script_path]

    # Linux / other POSIX. An explicit override wins so headless/custom setups
    # can point at any terminal: ZCU_AGENT_TERMINAL is the full program name and
    # the script path is appended as the final arg.
    override = os.environ.get("ZCU_AGENT_TERMINAL")
    if override:
        return [override, script_path]
    # gnome-terminal/konsole/xterm all run a shell with the script as a command;
    # the flag form differs per terminal.
    if shutil.which("gnome-terminal") is not None:
        return ["gnome-terminal", "--", "bash", script_path]
    if shutil.which("konsole") is not None:
        return ["konsole", "-e", "bash", script_path]
    if shutil.which("xterm") is not None:
        return ["xterm", "-e", "bash", script_path]
    if shutil.which("x-terminal-emulator") is not None:
        return ["x-terminal-emulator", "-e", "bash", script_path]
    raise RuntimeError(
        "No terminal emulator found (looked for gnome-terminal, konsole, xterm, "
        "x-terminal-emulator). Set the ZCU_AGENT_TERMINAL environment variable to "
        "your terminal program (it is invoked as '<terminal> <launch-script>')."
    )


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

    The spawn is detached (``subprocess.Popen``, not waited). The child env drops
    ``ANTHROPIC_API_KEY`` so claude uses subscription auth, mirroring the legacy
    runner. Cross-platform branches Fast-Fail rather than silently degrading.
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

    script_path = _write_launch_script(repo_root, argv)
    spawn_argv = _spawn_terminal_argv(script_path)

    env = os.environ.copy()
    # Subscription auth: claude must NOT see an API key (mirrors agent_runner).
    env.pop("ANTHROPIC_API_KEY", None)

    subprocess.Popen(spawn_argv, env=env)  # detached: do not wait
    return session_id
