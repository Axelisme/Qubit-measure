"""Tests for ``services.agent_launcher`` — the external-terminal claude launcher.

These exercise the Qt-free helpers in isolation (no real terminal is spawned;
``subprocess.Popen`` and ``shutil.which`` are monkeypatched). They cover argv
construction, the loopback MCP config, session-id format + persistence, and the
per-platform terminal-spawn branches including the Fast-Fail when no terminal
is found.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from zcu_tools.gui.app.main.services import agent_launcher

# ---------------------------------------------------------------------------
# build_claude_argv
# ---------------------------------------------------------------------------


def test_build_claude_argv_is_interactive_with_core_flags() -> None:
    argv = agent_launcher.build_claude_argv("/tmp/mcp.json")
    # Interactive mode: no print flag.
    assert "-p" not in argv
    assert "--output-format" not in argv
    # argv[0] is the configured agent command.
    assert argv[0] == agent_launcher.AGENT_CMD
    # Core flags present with their values.
    assert argv[argv.index("--mcp-config") + 1] == "/tmp/mcp.json"
    assert argv[argv.index("--allowedTools") + 1] == "mcp__measure-gui__*"
    assert "--append-system-prompt" in argv
    prompt = argv[argv.index("--append-system-prompt") + 1]
    assert "mcp__measure-gui__*" in prompt


def test_build_claude_argv_resume_branch() -> None:
    argv = agent_launcher.build_claude_argv(
        "/tmp/mcp.json", resume_session_id="sess-123"
    )
    assert argv[argv.index("--resume") + 1] == "sess-123"
    assert "--session-id" not in argv


def test_build_claude_argv_new_session_branch() -> None:
    argv = agent_launcher.build_claude_argv("/tmp/mcp.json", new_session_id="sess-new")
    assert argv[argv.index("--session-id") + 1] == "sess-new"
    assert "--resume" not in argv


def test_build_claude_argv_resume_wins_over_new() -> None:
    argv = agent_launcher.build_claude_argv(
        "/tmp/mcp.json", resume_session_id="r", new_session_id="n"
    )
    assert "--resume" in argv
    assert "--session-id" not in argv


def test_build_claude_argv_neither_session() -> None:
    argv = agent_launcher.build_claude_argv("/tmp/mcp.json")
    assert "--resume" not in argv
    assert "--session-id" not in argv


def test_build_claude_argv_respects_agent_cmd_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_launcher, "AGENT_CMD", "codex")
    argv = agent_launcher.build_claude_argv("/tmp/mcp.json")
    assert argv[0] == "codex"


# ---------------------------------------------------------------------------
# build_loopback_mcp_config
# ---------------------------------------------------------------------------


def test_build_loopback_mcp_config_points_at_measure_server() -> None:
    path = agent_launcher.build_loopback_mcp_config("/repo/root")
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    server = config["mcpServers"]["measure-gui"]
    assert server["command"] == "uv"
    assert server["args"] == [
        "run",
        "--extra",
        "gui",
        "python",
        "lib/zcu_tools/mcp/measure/server.py",
    ]
    assert server["cwd"] == "/repo/root"


# ---------------------------------------------------------------------------
# new_session_id
# ---------------------------------------------------------------------------


def test_new_session_id_is_dashed_uuid() -> None:
    sid = agent_launcher.new_session_id()
    # Round-trips through uuid.UUID and prints back to the same dashed string.
    assert str(uuid.UUID(sid)) == sid
    assert sid.count("-") == 4


# ---------------------------------------------------------------------------
# last-session file roundtrip
# ---------------------------------------------------------------------------


@pytest.fixture
def _last_session_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the last-session file into a temp dir."""
    path = tmp_path / "agent_last_session"
    monkeypatch.setattr(agent_launcher, "_LAST_SESSION_FILE", path)
    return path


def test_read_last_session_id_missing(_last_session_path: Path) -> None:
    assert agent_launcher.read_last_session_id() is None


def test_write_then_read_last_session_id(_last_session_path: Path) -> None:
    agent_launcher.write_last_session_id("abc-123")
    assert agent_launcher.read_last_session_id() == "abc-123"


def test_read_last_session_id_blank_is_none(_last_session_path: Path) -> None:
    _last_session_path.parent.mkdir(parents=True, exist_ok=True)
    _last_session_path.write_text("   \n", encoding="utf-8")
    assert agent_launcher.read_last_session_id() is None


def test_write_last_session_id_rejects_empty(_last_session_path: Path) -> None:
    with pytest.raises(ValueError):
        agent_launcher.write_last_session_id("")


# ---------------------------------------------------------------------------
# launch_agent_terminal
# ---------------------------------------------------------------------------


class _FakePopen:
    """Capture the argv + env a spawn would have used."""

    instances: list[_FakePopen] = []

    def __init__(self, argv: list[str], env: dict[str, str] | None = None) -> None:
        self.argv = argv
        self.env = env
        _FakePopen.instances.append(self)


@pytest.fixture(autouse=True)
def _reset_fake_popen() -> None:
    _FakePopen.instances = []


def _patch_linux_gnome(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_launcher.sys, "platform", "linux")
    monkeypatch.setattr(
        agent_launcher.shutil,
        "which",
        lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None,
    )
    monkeypatch.setattr(agent_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.delenv("ZCU_AGENT_TERMINAL", raising=False)


def test_launch_new_session_persists_and_spawns(
    monkeypatch: pytest.MonkeyPatch, _last_session_path: Path
) -> None:
    _patch_linux_gnome(monkeypatch)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "fixed-uuid")

    session_id = agent_launcher.launch_agent_terminal("/repo", resume=False)

    assert session_id == "fixed-uuid"
    assert agent_launcher.read_last_session_id() == "fixed-uuid"
    assert len(_FakePopen.instances) == 1
    spawn = _FakePopen.instances[0]
    # gnome-terminal runs the launch script via bash.
    assert spawn.argv[:3] == ["gnome-terminal", "--", "bash"]
    script_path = spawn.argv[3]
    script_body = Path(script_path).read_text(encoding="utf-8")
    # The script cds into the repo and execs claude in --session-id (new) mode.
    assert "cd /repo" in script_body
    assert "--session-id" in script_body
    assert "fixed-uuid" in script_body
    # ANTHROPIC_API_KEY is stripped for subscription auth.
    assert spawn.env is not None
    assert "ANTHROPIC_API_KEY" not in spawn.env


def test_launch_resume_uses_persisted_id(
    monkeypatch: pytest.MonkeyPatch, _last_session_path: Path
) -> None:
    _patch_linux_gnome(monkeypatch)
    agent_launcher.write_last_session_id("prev-sess")

    session_id = agent_launcher.launch_agent_terminal("/repo", resume=True)

    assert session_id == "prev-sess"
    script_body = Path(_FakePopen.instances[0].argv[3]).read_text(encoding="utf-8")
    assert "--resume" in script_body
    assert "prev-sess" in script_body


def test_launch_resume_with_no_last_falls_back_to_new(
    monkeypatch: pytest.MonkeyPatch, _last_session_path: Path
) -> None:
    _patch_linux_gnome(monkeypatch)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "fresh")

    session_id = agent_launcher.launch_agent_terminal("/repo", resume=True)

    assert session_id == "fresh"
    script_body = Path(_FakePopen.instances[0].argv[3]).read_text(encoding="utf-8")
    assert "--session-id" in script_body


def test_launch_terminal_override_env(
    monkeypatch: pytest.MonkeyPatch, _last_session_path: Path
) -> None:
    monkeypatch.setattr(agent_launcher.sys, "platform", "linux")
    monkeypatch.setattr(agent_launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(agent_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setenv("ZCU_AGENT_TERMINAL", "/opt/myterm")

    agent_launcher.launch_agent_terminal("/repo", resume=False)

    spawn = _FakePopen.instances[0]
    assert spawn.argv[0] == "/opt/myterm"
    assert Path(spawn.argv[1]).exists()


def test_launch_fast_fails_when_no_terminal(
    monkeypatch: pytest.MonkeyPatch, _last_session_path: Path
) -> None:
    monkeypatch.setattr(agent_launcher.sys, "platform", "linux")
    monkeypatch.setattr(agent_launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(agent_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.delenv("ZCU_AGENT_TERMINAL", raising=False)

    with pytest.raises(RuntimeError, match="ZCU_AGENT_TERMINAL"):
        agent_launcher.launch_agent_terminal("/repo", resume=False)
