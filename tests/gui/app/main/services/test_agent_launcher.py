"""Tests for ``services.agent_launcher`` — the external-terminal claude launcher.

These exercise the Qt-free helpers in isolation (no real terminal is spawned;
``subprocess.Popen`` and ``shutil.which`` are monkeypatched). They cover argv
construction, the loopback MCP config, session-id format, the session-list
store (record/dedup/cap), ``claude_project_dir`` slug encoding,
``list_resumable_sessions`` (label extraction / phantom-session skip / sorting /
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
    # argv[0] is the resolved agent command (see resolve_agent_command).
    assert argv[0] == agent_launcher.resolve_agent_command()
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
    # ZCU_AGENT_CMD is the explicit escape hatch — it wins on every platform.
    monkeypatch.setenv("ZCU_AGENT_CMD", "codex")
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
    # os.path.abspath prepends a drive letter on Windows (C:\home\user\myrepo →
    # "C--home-user-myrepo"), so assert the slug suffix rather than a hardcoded
    # POSIX result — the encoding of the path body is identical on both platforms.
    assert path.name.endswith("-home-user-myrepo")
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


def test_list_resumable_sessions_skips_phantom_without_jsonl(
    _sessions_env: tuple[Path, Path],
) -> None:
    # A recorded id whose jsonl was never created (a launch that Fast-Failed
    # before claude ran) is a phantom — there is nothing to resume, so it must
    # NOT be listed.
    sessions_file, _project_dir = _sessions_env
    sid = "bbbbbbbb-0000-0000-0000-000000000001"
    sessions_file.write_text(
        json.dumps([{"session_id": sid, "created": 2000.0}]), encoding="utf-8"
    )

    result = agent_launcher.list_resumable_sessions("/repo")
    assert result == []


def test_list_resumable_sessions_keeps_real_drops_phantom(
    _sessions_env: tuple[Path, Path],
) -> None:
    # Mixed store: only the id with a real jsonl is returned; the phantom is dropped.
    sessions_file, project_dir = _sessions_env
    real = "aaaaaaaa-0000-0000-0000-000000000009"
    phantom = "bbbbbbbb-0000-0000-0000-000000000009"
    (project_dir / f"{real}.jsonl").write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "real one"}})
        + "\n",
        encoding="utf-8",
    )
    sessions_file.write_text(
        json.dumps(
            [
                {"session_id": phantom, "created": 3000.0},
                {"session_id": real, "created": 1000.0},
            ]
        ),
        encoding="utf-8",
    )

    result = agent_launcher.list_resumable_sessions("/repo")
    assert [s.session_id for s in result] == [real]
    assert result[0].label == "real one"


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
    """Capture the argv + env + creationflags a spawn would have used."""

    instances: list[_FakePopen] = []

    def __init__(
        self,
        argv: list[str],
        env: dict[str, str] | None = None,
        creationflags: int = 0,
    ) -> None:
        self.argv = argv
        self.env = env
        self.creationflags = creationflags
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
# resolve_agent_command / _find_desktop_bundled_claude (Windows CLI resolution)
# ---------------------------------------------------------------------------


def test_resolve_agent_command_env_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The explicit override wins on every platform, ahead of any Desktop lookup.
    monkeypatch.setenv("ZCU_AGENT_CMD", "codex")
    monkeypatch.setattr(agent_launcher.sys, "platform", "win32")
    monkeypatch.setattr(
        agent_launcher, "_find_desktop_bundled_claude", lambda: r"C:\bundled\claude.exe"
    )
    assert agent_launcher.resolve_agent_command() == "codex"


def test_resolve_agent_command_non_windows_is_bare_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZCU_AGENT_CMD", raising=False)
    monkeypatch.setattr(agent_launcher.sys, "platform", "linux")
    monkeypatch.setattr(agent_launcher.shutil, "which", lambda name: None)
    # The Desktop lookup must not even run off Windows.
    monkeypatch.setattr(
        agent_launcher,
        "_find_desktop_bundled_claude",
        lambda: pytest.fail("must not probe Desktop off Windows"),
    )
    assert agent_launcher.resolve_agent_command() == "claude"


def test_resolve_agent_command_prefers_path_claude_over_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A standalone `claude` on PATH wins over the Desktop-bundled CLI (the latter
    # must not even be probed).
    monkeypatch.delenv("ZCU_AGENT_CMD", raising=False)
    monkeypatch.setattr(agent_launcher.sys, "platform", "win32")
    monkeypatch.setattr(
        agent_launcher.shutil, "which", lambda name: r"C:\Users\u\.local\bin\claude.exe"
    )
    monkeypatch.setattr(
        agent_launcher,
        "_find_desktop_bundled_claude",
        lambda: pytest.fail("must not probe Desktop when claude is on PATH"),
    )
    assert agent_launcher.resolve_agent_command() == "claude"


def test_resolve_agent_command_windows_uses_bundle_when_no_path_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No standalone `claude` on PATH → fall back to the Desktop-bundled CLI.
    monkeypatch.delenv("ZCU_AGENT_CMD", raising=False)
    monkeypatch.setattr(agent_launcher.sys, "platform", "win32")
    monkeypatch.setattr(agent_launcher.shutil, "which", lambda name: None)
    bundled = r"C:\Users\u\AppData\Roaming\Claude\claude-code\2.1.170\claude.exe"
    monkeypatch.setattr(agent_launcher, "_find_desktop_bundled_claude", lambda: bundled)
    assert agent_launcher.resolve_agent_command() == bundled


def test_resolve_agent_command_falls_back_to_bare_claude_when_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No PATH claude and no Desktop bundle → bare "claude" (the launcher then
    # Fast-Fails if it is absent).
    monkeypatch.delenv("ZCU_AGENT_CMD", raising=False)
    monkeypatch.setattr(agent_launcher.sys, "platform", "win32")
    monkeypatch.setattr(agent_launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(agent_launcher, "_find_desktop_bundled_claude", lambda: None)
    assert agent_launcher.resolve_agent_command() == "claude"


def test_find_desktop_bundled_claude_picks_newest_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Lay out %APPDATA%\Claude\claude-code\<ver>\claude.exe for several versions;
    # the highest *numeric* version must win (2.1.170 > 2.1.9, not lexicographic).
    base = tmp_path / "Claude" / "claude-code"
    for ver in ("2.1.9", "2.1.170", "2.0.300"):
        d = base / ver
        d.mkdir(parents=True)
        (d / "claude.exe").write_text("", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path))

    result = agent_launcher._find_desktop_bundled_claude()
    assert result == str(base / "2.1.170" / "claude.exe")


def test_find_desktop_bundled_claude_ignores_dirs_without_exe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A higher-version dir with no claude.exe must be skipped in favour of a
    # lower-version dir that does have one.
    base = tmp_path / "Claude" / "claude-code"
    (base / "9.9.9").mkdir(parents=True)
    have = base / "1.0.0"
    have.mkdir(parents=True)
    (have / "claude.exe").write_text("", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert agent_launcher._find_desktop_bundled_claude() == str(have / "claude.exe")


def test_find_desktop_bundled_claude_absent_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # APPDATA set but no Claude\claude-code tree → None (Desktop not installed).
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert agent_launcher._find_desktop_bundled_claude() is None


def test_find_desktop_bundled_claude_no_appdata_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APPDATA", raising=False)
    assert agent_launcher._find_desktop_bundled_claude() is None


# ---------------------------------------------------------------------------
# Windows terminal-spawn branch (CREATE_NEW_CONSOLE default; ZCU_AGENT_TERMINAL)
# ---------------------------------------------------------------------------


def _patch_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_launcher.sys, "platform", "win32")
    monkeypatch.setattr(agent_launcher.subprocess, "Popen", _FakePopen)
    # Pin agent-binary resolution + terminal override so the spawn is deterministic
    # regardless of host (these tests assert the terminal-spawn shape).
    monkeypatch.delenv("ZCU_AGENT_CMD", raising=False)
    monkeypatch.delenv("ZCU_AGENT_TERMINAL", raising=False)
    monkeypatch.setattr(agent_launcher, "_find_desktop_bundled_claude", lambda: None)


def test_launch_windows_default_runs_launcher_in_new_console(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_windows(monkeypatch)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "win-sess")

    agent_launcher.launch_agent_terminal("/repo")

    spawn = _FakePopen.instances[0]
    # The launcher is spawned DIRECTLY ([py, launcher.py]) — no ``wt`` (AppData
    # sandbox) and no ``cmd /c start`` (quoting pitfalls); the new window comes
    # from CREATE_NEW_CONSOLE, so subprocess quotes the two real paths itself.
    assert spawn.argv[0] == sys.executable
    launcher_path = spawn.argv[1]
    assert launcher_path.endswith(".py")
    assert "cmd" not in spawn.argv and "wt" not in spawn.argv
    # CREATE_NEW_CONSOLE is applied (resolved the same way the code does, so the
    # assertion holds on non-Windows hosts where the flag is absent → 0).
    assert spawn.creationflags == getattr(
        agent_launcher.subprocess, "CREATE_NEW_CONSOLE", 0
    )
    # The launcher still compiles on the Windows path.
    compile(Path(launcher_path).read_text(encoding="utf-8"), "<launcher>", "exec")


def test_launch_windows_terminal_override_wins(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    _patch_windows(monkeypatch)
    monkeypatch.setenv("ZCU_AGENT_TERMINAL", r"C:\tools\myterm.exe")
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "win-override")

    agent_launcher.launch_agent_terminal("/repo")

    spawn = _FakePopen.instances[0]
    # ZCU_AGENT_TERMINAL wins on Windows too: [<terminal>, <python>, <launcher.py>].
    # That terminal owns its own window, so CREATE_NEW_CONSOLE is NOT applied.
    assert spawn.argv[0] == r"C:\tools\myterm.exe"
    assert spawn.argv[1] == sys.executable
    assert spawn.argv[2].endswith(".py")
    assert spawn.creationflags == 0


def test_launch_windows_multiline_prompt_launcher_compiles(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    """#1 on Windows: multi-line state still produces a compilable launcher."""
    _patch_windows(monkeypatch)
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "win-ml")
    state = "[measure-gui current state]\nproject: chip=Q5\n[end state]"

    agent_launcher.launch_agent_terminal("/repo", state_context=state)

    spawn = _FakePopen.instances[0]
    # Direct spawn: the launcher path is argv[1].
    launcher_path = spawn.argv[1]
    source = Path(launcher_path).read_text(encoding="utf-8")
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


def test_strip_orchestration_env_removes_parent_vars() -> None:
    env = {
        "PATH": "x",
        "HOME": "h",
        "ANTHROPIC_API_KEY": "k",
        "CLAUDECODE": "1",
        "CLAUDE_CODE_ENTRYPOINT": "claude-desktop",
        "CLAUDE_CODE_SESSION_ID": "parent",
        "CLAUDE_AGENT_SDK_VERSION": "0.3.170",
    }
    out = agent_launcher._strip_orchestration_env(env)
    assert out is env  # mutated in place
    # Only the parent's Claude Code orchestration vars are removed.
    assert set(env) == {"PATH", "HOME"}


def test_launch_strips_orchestration_env_from_spawn(
    monkeypatch: pytest.MonkeyPatch, _sessions_file: Path
) -> None:
    # The bug: the child claude inherited CLAUDE_CODE_ENTRYPOINT=claude-desktop
    # (and friends) and started Desktop-embedded, injecting a phantom "are".
    _patch_linux_gnome(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "claude-desktop")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_AGENT_SDK_VERSION", "0.3.170")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-sess")
    monkeypatch.setenv("ZCU_KEEP_ME", "yes")  # a normal var must survive
    monkeypatch.setattr(agent_launcher, "new_session_id", lambda: "s")

    agent_launcher.launch_agent_terminal("/repo")

    spawn = _FakePopen.instances[0]
    assert spawn.env is not None
    for k in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDECODE",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_CODE_SESSION_ID",
    ):
        assert k not in spawn.env
    assert spawn.env.get("ZCU_KEEP_ME") == "yes"
    # The launcher source strips them at runtime too (double safety).
    source = _spawned_launcher_source()
    assert "_STRIP_PREFIXES" in source
    assert "CLAUDE_CODE_" in source


def test_launcher_source_strips_orchestration_env() -> None:
    source = agent_launcher.build_python_launcher_source("/repo", ["claude"])
    compile(source, "<launcher>", "exec")
    assert "_STRIP_EXACT" in source and "_STRIP_PREFIXES" in source
    assert "ANTHROPIC_API_KEY" in source
    assert "CLAUDECODE" in source
    assert "CLAUDE_CODE_" in source
    assert "CLAUDE_AGENT_SDK" in source


def test_launcher_resolves_binary_via_which(monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher source resolves argv[0] via shutil.which (Windows .cmd)."""
    # This is a source-level assertion: the launcher does ``shutil.which(ARGV[0])``
    # so a Windows ``claude.cmd`` on PATH resolves even though os.execv would not.
    source = agent_launcher.build_python_launcher_source("/repo", ["claude"])
    assert "shutil.which(ARGV[0])" in source
    assert "os.execv(_bin, ARGV)" in source
