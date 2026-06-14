"""Tests for ``services.agent_launcher`` — the external-terminal claude launcher.

These exercise the Qt-free helpers in isolation (no real terminal is spawned;
``subprocess.Popen`` and ``shutil.which`` are monkeypatched). They cover argv
construction, the loopback MCP config, session-id format, the session-list
store (record/dedup/cap), ``claude_project_dir`` slug encoding,
``list_resumable_sessions`` (label extraction / jsonl fallback / sorting /
empty store), the cross-platform Python launcher (json-embedded argv/cwd that
keeps a multi-line prompt safe — ``compile`` regression), and the per-platform
terminal-spawn branches (Linux / Windows-with-wt / Windows-without-wt) including
the Fast-Fail when no terminal is found.
"""

from __future__ import annotations

import json
import sys
import time
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


def test_build_claude_argv_appends_state_context_to_system_prompt() -> None:
    state = "[measure-gui current state]\nproject: chip=C\n[end state]"
    argv = agent_launcher.build_claude_argv("/tmp/mcp.json", state_context=state)
    prompt = argv[argv.index("--append-system-prompt") + 1]
    # The static embedded prompt and the state block both ride one flag, with the
    # state block appended after the embedded prompt (Round 2 state injection).
    assert agent_launcher._EMBEDDED_SYSTEM_PROMPT in prompt
    assert prompt.endswith(state)
    assert "[measure-gui current state]" in prompt


def test_build_claude_argv_without_state_context_is_static_prompt_only() -> None:
    argv = agent_launcher.build_claude_argv("/tmp/mcp.json")
    prompt = argv[argv.index("--append-system-prompt") + 1]
    # Default (no state_context): only the static embedded prompt, no state block.
    assert prompt == agent_launcher._EMBEDDED_SYSTEM_PROMPT
    assert "[measure-gui current state]" not in prompt


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
# Session store: record_launched_session
# ---------------------------------------------------------------------------


@pytest.fixture
def _sessions_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect _SESSIONS_FILE into a temp dir."""
    path = tmp_path / "agent_sessions.json"
    monkeypatch.setattr(agent_launcher, "_SESSIONS_FILE", path)
    return path


def test_record_launched_session_creates_file(_sessions_file: Path) -> None:
    agent_launcher.record_launched_session("abc-123")
    records = json.loads(_sessions_file.read_text())
    assert len(records) == 1
    assert records[0]["session_id"] == "abc-123"
    assert "created" in records[0]


def test_record_launched_session_appends_newest_first(
    _sessions_file: Path,
) -> None:
    agent_launcher.record_launched_session("first")
    agent_launcher.record_launched_session("second")
    records = json.loads(_sessions_file.read_text())
    assert records[0]["session_id"] == "second"
    assert records[1]["session_id"] == "first"


def test_record_launched_session_deduplicates(_sessions_file: Path) -> None:
    agent_launcher.record_launched_session("dup")
    agent_launcher.record_launched_session("other")
    agent_launcher.record_launched_session("dup")  # moves dup to front
    records = json.loads(_sessions_file.read_text())
    ids = [r["session_id"] for r in records]
    assert ids.count("dup") == 1
    assert ids[0] == "dup"


def test_record_launched_session_caps_at_limit(
    _sessions_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_launcher, "_SESSION_CAP", 3)
    for i in range(5):
        agent_launcher.record_launched_session(f"sess-{i}")
    records = json.loads(_sessions_file.read_text())
    assert len(records) == 3
    # Most recently recorded are kept.
    assert records[0]["session_id"] == "sess-4"


def test_record_launched_session_rejects_empty(_sessions_file: Path) -> None:
    with pytest.raises(ValueError):
        agent_launcher.record_launched_session("")


# ---------------------------------------------------------------------------
# claude_project_dir slug encoding
# ---------------------------------------------------------------------------


def test_claude_project_dir_simple_path() -> None:
    path = agent_launcher.claude_project_dir("/home/user/myrepo")
    assert path.name == "-home-user-myrepo"
    assert path.parent.name == "projects"


def test_claude_project_dir_special_chars() -> None:
    # Dots and underscores become dashes; hyphens are preserved.
    path = agent_launcher.claude_project_dir("/home/user/my.repo_v2")
    assert "-" in path.name
    assert "." not in path.name
    assert "_" not in path.name


def test_claude_project_dir_dotclaude_segment() -> None:
    # The ".claude" segment in a path must encode the dot.
    path = agent_launcher.claude_project_dir("/home/user/.claude/repo")
    # ".claude" → "--claude" (dot becomes dash).
    assert "--claude" in path.name


# ---------------------------------------------------------------------------
# list_resumable_sessions
# ---------------------------------------------------------------------------


@pytest.fixture
def _sessions_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Redirect _SESSIONS_FILE and claude_project_dir into tmp_path.

    Returns (sessions_file, project_dir).
    """
    sessions_file = tmp_path / "agent_sessions.json"
    monkeypatch.setattr(agent_launcher, "_SESSIONS_FILE", sessions_file)

    project_dir = tmp_path / "claude_project"
    project_dir.mkdir()

    # Override claude_project_dir to return our temp project_dir.
    monkeypatch.setattr(agent_launcher, "claude_project_dir", lambda _root: project_dir)
    return sessions_file, project_dir


def test_list_resumable_sessions_empty_store(
    _sessions_env: tuple[Path, Path],
) -> None:
    sessions = agent_launcher.list_resumable_sessions("/repo")
    assert sessions == []


def test_list_resumable_sessions_returns_recorded_sessions(
    _sessions_env: tuple[Path, Path],
) -> None:
    sessions_file, project_dir = _sessions_env
    sid = "aaaaaaaa-0000-0000-0000-000000000001"
    # Write a minimal jsonl with a user message.
    jsonl = project_dir / f"{sid}.jsonl"
    jsonl.write_text(
        json.dumps(
            {"type": "user", "message": {"role": "user", "content": "Hello there!"}}
        )
        + "\n",
        encoding="utf-8",
    )
    sessions_file.write_text(
        json.dumps([{"session_id": sid, "created": 1000.0}]), encoding="utf-8"
    )

    result = agent_launcher.list_resumable_sessions("/repo")
    assert len(result) == 1
    assert result[0].session_id == sid
    assert result[0].label == "Hello there!"


def test_list_resumable_sessions_label_truncated(
    _sessions_env: tuple[Path, Path],
) -> None:
    sessions_file, project_dir = _sessions_env
    sid = "aaaaaaaa-0000-0000-0000-000000000002"
    long_text = "A" * 100
    jsonl = project_dir / f"{sid}.jsonl"
    jsonl.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": long_text}})
        + "\n",
        encoding="utf-8",
    )
    sessions_file.write_text(
        json.dumps([{"session_id": sid, "created": 1000.0}]), encoding="utf-8"
    )

    result = agent_launcher.list_resumable_sessions("/repo")
    assert len(result[0].label) == agent_launcher._LABEL_MAX_CHARS


def test_list_resumable_sessions_jsonl_missing_fallback(
    _sessions_env: tuple[Path, Path],
) -> None:
    sessions_file, _project_dir = _sessions_env
    sid = "bbbbbbbb-0000-0000-0000-000000000001"
    sessions_file.write_text(
        json.dumps([{"session_id": sid, "created": 2000.0}]), encoding="utf-8"
    )

    result = agent_launcher.list_resumable_sessions("/repo")
    assert len(result) == 1
    # Label falls back to first 8 chars of sid.
    assert result[0].label == sid[:8]
    # last_active falls back to stored created timestamp.
    assert result[0].last_active == pytest.approx(2000.0)


def test_list_resumable_sessions_malformed_lines_do_not_crash(
    _sessions_env: tuple[Path, Path],
) -> None:
    sessions_file, project_dir = _sessions_env
    sid = "cccccccc-0000-0000-0000-000000000001"
    jsonl = project_dir / f"{sid}.jsonl"
    # Mix of malformed JSON, wrong types, and a valid user message.
    lines = [
        "not-json\n",
        json.dumps({"type": "user", "message": None}) + "\n",  # None message
        json.dumps({"type": "user", "message": {"role": "user", "content": ""}})
        + "\n",  # empty
        json.dumps({"type": "user", "message": {"role": "user", "content": "Good msg"}})
        + "\n",
    ]
    jsonl.write_text("".join(lines), encoding="utf-8")
    sessions_file.write_text(
        json.dumps([{"session_id": sid, "created": 1000.0}]), encoding="utf-8"
    )

    # Must not raise; label should be the first parseable non-empty user text.
    result = agent_launcher.list_resumable_sessions("/repo")
    assert result[0].label == "Good msg"


def test_list_resumable_sessions_content_as_list(
    _sessions_env: tuple[Path, Path],
) -> None:
    """content as a list of blocks — pick the first text block."""
    sessions_file, project_dir = _sessions_env
    sid = "dddddddd-0000-0000-0000-000000000001"
    jsonl = project_dir / f"{sid}.jsonl"
    content = [{"type": "text", "text": "Block message"}]
    jsonl.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": content}})
        + "\n",
        encoding="utf-8",
    )
    sessions_file.write_text(
        json.dumps([{"session_id": sid, "created": 1000.0}]), encoding="utf-8"
    )

    result = agent_launcher.list_resumable_sessions("/repo")
    assert result[0].label == "Block message"


def test_list_resumable_sessions_sorted_newest_first(
    _sessions_env: tuple[Path, Path],
) -> None:
    sessions_file, project_dir = _sessions_env
    now = time.time()
    # Two sessions: older jsonl mtime vs newer.
    sid_old = "eeeeeeee-0000-0000-0000-000000000001"
    sid_new = "eeeeeeee-0000-0000-0000-000000000002"

    jsonl_old = project_dir / f"{sid_old}.jsonl"
    jsonl_new = project_dir / f"{sid_new}.jsonl"
    jsonl_old.write_text("{}\n", encoding="utf-8")
    jsonl_new.write_text("{}\n", encoding="utf-8")
    # Set mtimes explicitly.
    import os

    os.utime(jsonl_old, (now - 3600, now - 3600))
    os.utime(jsonl_new, (now, now))

    sessions_file.write_text(
        json.dumps(
            [
                {"session_id": sid_old, "created": now - 3600},
                {"session_id": sid_new, "created": now},
            ]
        ),
        encoding="utf-8",
    )

    result = agent_launcher.list_resumable_sessions("/repo")
    assert result[0].session_id == sid_new
    assert result[1].session_id == sid_old


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


def _spawned_launcher_source() -> str:
    """Read the launcher ``.py`` from the single recorded spawn.

    The terminal command is always ``[<term>..., python, launcher_path]`` and the
    launcher path is the *last* argv token, so reading argv[-1] works across the
    gnome / konsole / xterm / override branches.
    """
    launcher_path = _FakePopen.instances[0].argv[-1]
    return Path(launcher_path).read_text(encoding="utf-8")


def test_launch_new_session_records_and_spawns(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_linux_gnome(monkeypatch)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "fixed-uuid")

    session_id = agent_launcher.launch_agent_terminal("/repo")

    assert session_id == "fixed-uuid"
    # Session must be recorded in the store.
    records = json.loads(_sessions_file.read_text())
    assert any(r["session_id"] == "fixed-uuid" for r in records)
    assert len(_FakePopen.instances) == 1
    spawn = _FakePopen.instances[0]
    # gnome-terminal runs the python launcher: [term, --, <python>, <launcher.py>].
    assert spawn.argv[:2] == ["gnome-terminal", "--"]
    assert spawn.argv[2] == sys.executable
    launcher_path = spawn.argv[3]
    assert launcher_path.endswith(".py")
    source = Path(launcher_path).read_text(encoding="utf-8")
    # The launcher chdirs into the repo and execs claude in --session-id mode.
    # repo_root is abspath'd, so match the basename rather than the literal "/repo".
    assert "os.chdir(CWD)" in source
    assert '"--session-id"' in source
    assert "fixed-uuid" in source
    # ANTHROPIC_API_KEY is stripped for subscription auth (both in env and launcher).
    assert spawn.env is not None
    assert "ANTHROPIC_API_KEY" not in spawn.env
    assert "ANTHROPIC_API_KEY" in source  # the launcher also pops it


def test_launch_resume_uses_given_session_id(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_linux_gnome(monkeypatch)

    session_id = agent_launcher.launch_agent_terminal(
        "/repo", resume_session_id="prev-sess"
    )

    assert session_id == "prev-sess"
    source = _spawned_launcher_source()
    assert '"--resume"' in source
    assert "prev-sess" in source


def test_launch_resume_does_not_re_record(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    """Resuming an existing session must not add a duplicate record."""
    _patch_linux_gnome(monkeypatch)
    # Pre-populate the store with the session being resumed.
    _sessions_file.parent.mkdir(parents=True, exist_ok=True)
    _sessions_file.write_text(
        json.dumps([{"session_id": "prev-sess", "created": 1000.0}]), encoding="utf-8"
    )

    agent_launcher.launch_agent_terminal("/repo", resume_session_id="prev-sess")

    records = json.loads(_sessions_file.read_text())
    # Still exactly one entry (no re-record on resume).
    assert len(records) == 1


def test_launch_passes_state_context_into_launcher(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_linux_gnome(monkeypatch)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "sess-state")
    # Multi-line state context: the very case that broke a .bat (#1). It is
    # json-embedded into the launcher, so it must round-trip and still compile.
    state = "[measure-gui current state]\nproject: chip=Q5\nopen tabs: 3\n[end state]"

    agent_launcher.launch_agent_terminal("/repo", state_context=state)

    source = _spawned_launcher_source()
    assert "measure-gui current state" in source
    assert "chip=Q5" in source
    # The multi-line value must not break the generated launcher's syntax.
    compile(source, "<launcher>", "exec")


def test_launch_without_state_context_omits_state_block(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_linux_gnome(monkeypatch)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "sess-plain")

    agent_launcher.launch_agent_terminal("/repo")

    source = _spawned_launcher_source()
    assert "measure-gui current state" not in source


def test_launch_terminal_override_env(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    monkeypatch.setattr(agent_launcher.sys, "platform", "linux")
    monkeypatch.setattr(agent_launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(agent_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.setenv("ZCU_AGENT_TERMINAL", "/opt/myterm")

    agent_launcher.launch_agent_terminal("/repo")

    spawn = _FakePopen.instances[0]
    # Override branch: [<terminal>, <python>, <launcher.py>].
    assert spawn.argv[0] == "/opt/myterm"
    assert spawn.argv[1] == sys.executable
    assert Path(spawn.argv[2]).exists()
    assert spawn.argv[2].endswith(".py")


def test_launch_fast_fails_when_no_terminal(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    monkeypatch.setattr(agent_launcher.sys, "platform", "linux")
    monkeypatch.setattr(agent_launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(agent_launcher.subprocess, "Popen", _FakePopen)
    monkeypatch.delenv("ZCU_AGENT_TERMINAL", raising=False)

    with pytest.raises(RuntimeError, match="ZCU_AGENT_TERMINAL"):
        agent_launcher.launch_agent_terminal("/repo")


# ---------------------------------------------------------------------------
# build_python_launcher_source — the cross-platform launcher (#1 regression)
# ---------------------------------------------------------------------------


def test_launcher_source_compiles_and_embeds_argv_and_cwd() -> None:
    argv = ["claude", "--allowedTools", "mcp__measure-gui__*"]
    source = agent_launcher.build_python_launcher_source("/repo/root", argv)
    # Must be valid Python.
    compile(source, "<launcher>", "exec")
    # argv/cwd are embedded as json (valid Python literals).
    assert json.dumps(argv) in source
    import os as _os

    assert json.dumps(_os.path.abspath("/repo/root")) in source
    # Resolves the binary via shutil.which and execs it.
    assert "shutil.which(ARGV[0])" in source
    assert "os.execv(_bin, ARGV)" in source
    # Drops the API key for subscription auth.
    assert "ANTHROPIC_API_KEY" in source


def test_launcher_source_multiline_prompt_does_not_break_syntax() -> None:
    """#1 regression: a multi-line --append-system-prompt must stay safe.

    A multi-line GUI-state snapshot broke .bat quoting; json-embedding it into a
    Python launcher must keep the source compilable AND preserve the full prompt.
    """
    multiline_prompt = (
        "You are operating a measure-gui.\n"
        "[measure-gui current state]\n"
        "project: chip=\"Q5\" qub='A'\n"
        "open tabs: 3\n"
        "weird chars: $ ` \\ % ! ^ & |\n"
        "[end state]"
    )
    argv = ["claude", "--append-system-prompt", multiline_prompt]
    source = agent_launcher.build_python_launcher_source("/repo", argv)

    # Compiles despite newlines / quotes / backslashes in the prompt.
    code = compile(source, "<launcher>", "exec")
    # Execute just the assignment lines to recover ARGV and assert the full
    # multi-line prompt round-trips intact (json literal preserves everything).
    namespace: dict[str, object] = {}
    # ARGV/CWD are plain literal assignments; exec'ing the first 3 lines is safe
    # (import + ARGV + CWD), the os.chdir/execv lines are not reached.
    header = "\n".join(source.splitlines()[:3])
    exec(compile(header, "<launcher-header>", "exec"), namespace)
    assert namespace["ARGV"] == argv
    assert multiline_prompt in namespace["ARGV"]  # type: ignore[operator]
    assert code is not None


def test_launcher_source_missing_binary_exits_with_message() -> None:
    source = agent_launcher.build_python_launcher_source("/repo", ["claude"])
    # The Fast-Fail path teaches the user about ZCU_AGENT_CMD.
    assert "command not found on PATH" in source
    assert "ZCU_AGENT_CMD" in source


# ---------------------------------------------------------------------------
# Windows terminal-spawn branches (#2 wt, #4 cmd start quoting)
# ---------------------------------------------------------------------------


def _patch_windows(monkeypatch: pytest.MonkeyPatch, *, has_wt: bool) -> None:
    monkeypatch.setattr(agent_launcher.sys, "platform", "win32")

    def _which(name: str) -> str | None:
        if name == "wt" and has_wt:
            return r"C:\Windows\System32\wt.exe"
        return None

    monkeypatch.setattr(agent_launcher.shutil, "which", _which)
    monkeypatch.setattr(agent_launcher.subprocess, "Popen", _FakePopen)


def test_launch_windows_with_wt_uses_new_tab(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_windows(monkeypatch, has_wt=True)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "win-sess")

    agent_launcher.launch_agent_terminal("/repo")

    spawn = _FakePopen.instances[0]
    # #2 fix: ``wt new-tab <python> <launcher>`` runs an external command in a new
    # tab (the trailing tokens are wt's ``commandline``), not "open a cmd profile".
    assert spawn.argv[0] == r"C:\Windows\System32\wt.exe"
    assert spawn.argv[1] == "new-tab"
    assert spawn.argv[2] == sys.executable
    assert spawn.argv[3].endswith(".py")
    # The launcher still compiles on the Windows path.
    compile(Path(spawn.argv[3]).read_text(encoding="utf-8"), "<launcher>", "exec")


def test_launch_windows_without_wt_uses_cmd_start_with_quotes(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_windows(monkeypatch, has_wt=False)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "win-sess2")

    agent_launcher.launch_agent_terminal("/repo")

    spawn = _FakePopen.instances[0]
    # #4 fix: ``cmd /c start "" "<py>" "<launcher>"`` — the empty "" is start's
    # window title, and BOTH paths are double-quoted so spaces do not split them.
    assert spawn.argv[:4] == ["cmd", "/c", "start", ""]
    quoted_py = spawn.argv[4]
    quoted_launcher = spawn.argv[5]
    assert quoted_py.startswith('"') and quoted_py.endswith('"')
    assert quoted_launcher.startswith('"') and quoted_launcher.endswith('"')
    assert sys.executable in quoted_py
    assert quoted_launcher.strip('"').endswith(".py")


def test_launch_windows_multiline_prompt_launcher_compiles(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    """#1 on Windows: multi-line state still produces a compilable launcher."""
    _patch_windows(monkeypatch, has_wt=True)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "win-ml")
    state = "[measure-gui current state]\nproject: chip=Q5\n[end state]"

    agent_launcher.launch_agent_terminal("/repo", state_context=state)

    spawn = _FakePopen.instances[0]
    source = Path(spawn.argv[3]).read_text(encoding="utf-8")
    compile(source, "<launcher>", "exec")
    assert "measure-gui current state" in source


# ---------------------------------------------------------------------------
# Binary resolution + env handling
# ---------------------------------------------------------------------------


def test_launch_drops_api_key_from_spawn_env(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_linux_gnome(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "sess-env")

    agent_launcher.launch_agent_terminal("/repo")

    spawn = _FakePopen.instances[0]
    assert spawn.env is not None
    assert "ANTHROPIC_API_KEY" not in spawn.env


def test_launcher_resolves_binary_via_which(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher source resolves argv[0] via shutil.which (Windows .cmd)."""
    # This is a source-level assertion: the launcher does ``shutil.which(ARGV[0])``
    # so a Windows ``claude.cmd`` on PATH resolves even though os.execv would not.
    source = agent_launcher.build_python_launcher_source(
        "/repo", [agent_launcher.AGENT_CMD]
    )
    assert "shutil.which(ARGV[0])" in source
    assert "os.execv(_bin, ARGV)" in source
