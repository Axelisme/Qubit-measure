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
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

# argv[0] for the agent CLI. ``claude`` by default; ``ZCU_AGENT_CMD`` overrides
# it so a future codex-style CLI can be swapped in without code changes.
AGENT_CMD: str = os.environ.get("ZCU_AGENT_CMD", "claude")

# Persisted "last session" file. A single id is enough — Round 1 supports one
# agent terminal at a time and only the "resume last" affordance reads it.
_LAST_SESSION_FILE = Path.home() / ".cache" / "zcu-tools" / "agent_last_session"

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


def read_last_session_id() -> str | None:
    """Return the persisted last-launched session id, or None if unset.

    Returns None when the file is missing or empty (whitespace-only). A blank
    file is treated as "no session" rather than an error so a corrupted/cleared
    cache degrades to the New-session path.
    """
    try:
        text = _LAST_SESSION_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return text or None


def write_last_session_id(session_id: str) -> None:
    """Persist ``session_id`` as the last-launched session (atomic replace).

    Writes to a sibling temp file then ``os.replace`` so a concurrent reader
    never sees a half-written id. No flock — Round 1 has at most one writer.
    """
    if not session_id:
        raise ValueError("session_id must be non-empty")
    _LAST_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _LAST_SESSION_FILE.with_suffix(".tmp")
    tmp.write_text(session_id, encoding="utf-8")
    os.replace(tmp, _LAST_SESSION_FILE)


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


def launch_agent_terminal(
    repo_root: str, *, resume: bool, state_context: str | None = None
) -> str:
    """Open the system terminal running interactive ``claude`` against the GUI.

    Decides the session (resume the persisted last id when ``resume`` is True and
    one exists; otherwise a fresh uuid that is persisted as the new last id),
    builds the loopback MCP config + argv, writes a launcher script, and spawns a
    detached terminal emulator running it. Returns the session id that was used.

    ``state_context`` (a live GUI-state snapshot from the Controller) is appended
    to the embedded system prompt for both the new and resume paths so the agent
    starts already knowing the current project / context / SoC / open tabs
    (Round 2). ``None`` keeps only the static embedded prompt.

    The spawn is detached (``subprocess.Popen``, not waited). The child env drops
    ``ANTHROPIC_API_KEY`` so claude uses subscription auth, mirroring the legacy
    runner. Cross-platform branches Fast-Fail rather than silently degrading.
    """
    mcp_config_path = build_loopback_mcp_config(repo_root)

    last = read_last_session_id() if resume else None
    if last is not None:
        session_id = last
        argv = build_claude_argv(
            mcp_config_path,
            resume_session_id=session_id,
            state_context=state_context,
        )
    else:
        session_id = new_session_id()
        write_last_session_id(session_id)
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
