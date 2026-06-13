"""Tests for agent_supervisor pure-logic components.

No real claude process is spawned.  All tests exercise the spool/log helpers
and argument-building functions in isolation, plus platform-dispatch mocking
for the detached spawn and stop helpers.

Coverage goals
--------------
  S1  - spool filename format is sortable and has correct extension.
  S2  - write_spool_message produces valid JSON spool file and renames atomically.
  S3  - consume_spool_entries returns files in name-sorted order.
  S4  - consume_spool_entries skips .tmp files and invalid JSON.
  S5  - consume_spool_entries skips partially-written (invalid JSON) files.
  S6  - append_log_line writes a newline-terminated line and is idempotent.
  S7  - argv built by build_claude_argv contains expected flags (smoke).
  S8  - spawn_supervisor_detached uses start_new_session=True on POSIX.
  S9  - spawn_supervisor_detached uses correct creationflags on Windows.
  S10 - stop_supervisor calls os.kill(SIGINT) on POSIX.
  S11 - stop_supervisor calls CTRL_BREAK_EVENT on Windows, falls back to taskkill.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# S1 — spool filename format
# ---------------------------------------------------------------------------


def test_spool_filename_has_json_extension() -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import _spool_filename

    name = _spool_filename(time.time())
    assert name.endswith(".json")


def test_spool_filename_is_sortable_by_timestamp() -> None:
    """Later timestamps must sort after earlier ones."""
    from zcu_tools.gui.app.main.services.agent_supervisor import _spool_filename

    t1 = _spool_filename(1000.0)
    t2 = _spool_filename(2000.0)
    assert t1 < t2


def test_spool_filename_unique_same_timestamp() -> None:
    """Two filenames with the same timestamp must differ (random suffix)."""
    from zcu_tools.gui.app.main.services.agent_supervisor import _spool_filename

    names = {_spool_filename(1234.0) for _ in range(20)}
    assert len(names) > 1


# ---------------------------------------------------------------------------
# S2 — write_spool_message
# ---------------------------------------------------------------------------


def test_write_spool_message_creates_valid_json(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import write_spool_message

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    written = write_spool_message(spool_dir, "hello agent")
    assert written.exists()
    assert written.suffix == ".json"
    obj = json.loads(written.read_bytes())
    # Content must be the stream-json stdin envelope.
    assert obj["type"] == "user"
    assert obj["message"]["content"] == "hello agent"


def test_write_spool_message_no_tmp_file_remains(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import write_spool_message

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    write_spool_message(spool_dir, "test")
    tmp_files = list(spool_dir.glob("*.tmp"))
    assert tmp_files == [], "temporary .tmp file must be cleaned up after rename"


# ---------------------------------------------------------------------------
# S3 & S4 & S5 — consume_spool_entries
# ---------------------------------------------------------------------------


def test_consume_spool_entries_sorted_fifo(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import (
        consume_spool_entries,
        write_spool_message,
    )

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    # Write messages with slightly different timestamps via fixed names.
    write_spool_message(spool_dir, "first")
    time.sleep(0.002)  # ensure ms-level timestamp difference
    write_spool_message(spool_dir, "second")

    entries = consume_spool_entries(spool_dir)
    assert len(entries) == 2
    # Filenames must be in ascending (FIFO) order.
    assert entries[0].name < entries[1].name


def test_consume_spool_entries_skips_tmp_files(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import consume_spool_entries

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    # Create a .tmp file (incomplete write simulation).
    (spool_dir / "0000000000001_abcdefgh.json.tmp").write_text('{"type":"user"}')
    entries = consume_spool_entries(spool_dir)
    assert entries == []


def test_consume_spool_entries_skips_invalid_json(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import consume_spool_entries

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    # Simulate a partially-written spool file (invalid JSON).
    (spool_dir / "0000000000001_abcdefgh.json").write_bytes(b'{"type":')
    entries = consume_spool_entries(spool_dir)
    assert entries == []


def test_consume_spool_entries_returns_valid_only(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import (
        consume_spool_entries,
        write_spool_message,
    )

    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    good = write_spool_message(spool_dir, "good")
    (spool_dir / "0000000000000_badinput.json").write_bytes(b"not json")
    entries = consume_spool_entries(spool_dir)
    assert len(entries) == 1
    assert entries[0] == good


# ---------------------------------------------------------------------------
# S6 — append_log_line
# ---------------------------------------------------------------------------


def test_append_log_line_creates_file(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import append_log_line

    log_path = tmp_path / "log.ndjson"
    append_log_line(log_path, '{"type":"system"}')
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert lines == ['{"type":"system"}']


def test_append_log_line_multiple_lines(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import append_log_line

    log_path = tmp_path / "log.ndjson"
    append_log_line(log_path, '{"type":"system"}')
    append_log_line(log_path, '{"type":"assistant"}')
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0] == '{"type":"system"}'
    assert lines[1] == '{"type":"assistant"}'


def test_append_log_line_strips_extra_newline(tmp_path: Path) -> None:
    """Lines from claude stdout may already have a trailing newline; strip it."""
    from zcu_tools.gui.app.main.services.agent_supervisor import append_log_line

    log_path = tmp_path / "log.ndjson"
    append_log_line(log_path, '{"type":"system"}\n')
    lines = log_path.read_text().splitlines()
    assert lines == ['{"type":"system"}']


# ---------------------------------------------------------------------------
# S7 — argv smoke (delegates to build_claude_argv already tested in test_agent_runner)
# ---------------------------------------------------------------------------


def test_supervisor_argv_reuses_build_claude_argv() -> None:
    """The supervisor invokes the same argv builder used by AgentRunner."""
    from zcu_tools.gui.app.main.services.agent_runner import build_claude_argv

    argv = build_claude_argv("task", "/tmp/mcp.json")
    assert "claude" in argv[0]
    assert "--output-format" in argv
    assert "stream-json" in argv


# ---------------------------------------------------------------------------
# S8 — spawn_supervisor_detached: POSIX start_new_session
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
def test_spawn_supervisor_posix_uses_start_new_session(tmp_path: Path) -> None:
    """On POSIX, Popen must be called with start_new_session=True."""
    from zcu_tools.gui.app.main.services.agent_supervisor import (
        spawn_supervisor_detached,
    )

    mock_proc = MagicMock()
    mock_proc.pid = 99999

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        handle = spawn_supervisor_detached(tmp_path, "test task", "/repo")

    assert mock_popen.called
    _, kwargs = mock_popen.call_args
    assert kwargs.get("start_new_session") is True, (
        "POSIX detached spawn must set start_new_session=True"
    )
    assert handle.pid == 99999
    assert handle.log_path.name == "log.ndjson"
    assert handle.spool_dir.name == "spool"


# ---------------------------------------------------------------------------
# S9 — spawn_supervisor_detached: Windows creationflags
# ---------------------------------------------------------------------------


def test_spawn_supervisor_windows_uses_detached_creationflags(
    tmp_path: Path,
) -> None:
    """On Windows (mocked), Popen must use DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP."""
    from zcu_tools.gui.app.main.services import agent_supervisor as mod

    mock_proc = MagicMock()
    mock_proc.pid = 12345

    # Simulate subprocess module having Windows-specific attributes.
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    expected_flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    with (
        patch.object(mod, "sys") as mock_sys,
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
        patch("subprocess.DETACHED_PROCESS", DETACHED_PROCESS, create=True),
        patch(
            "subprocess.CREATE_NEW_PROCESS_GROUP", CREATE_NEW_PROCESS_GROUP, create=True
        ),
    ):
        mock_sys.platform = "win32"
        mock_sys.executable = sys.executable
        # Call via the module reference so the patched sys.platform is seen.
        handle = mod.spawn_supervisor_detached(tmp_path, "test task", "/repo")

    assert mock_popen.called
    _, kwargs = mock_popen.call_args
    assert kwargs.get("creationflags") == expected_flags, (
        f"Expected creationflags={expected_flags:#x}, got {kwargs.get('creationflags')}"
    )
    assert handle.pid == 12345


# ---------------------------------------------------------------------------
# S10 — stop_supervisor: POSIX SIGINT
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
def test_stop_supervisor_posix_sends_sigint() -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import stop_supervisor

    with patch("os.kill") as mock_kill:
        stop_supervisor(54321)
    mock_kill.assert_called_once_with(54321, signal.SIGINT)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
def test_stop_supervisor_posix_tolerates_missing_process() -> None:
    """ProcessLookupError (already dead) must not raise."""
    from zcu_tools.gui.app.main.services.agent_supervisor import stop_supervisor

    with patch("os.kill", side_effect=ProcessLookupError):
        stop_supervisor(99999)  # must not raise


# ---------------------------------------------------------------------------
# S11 — stop_supervisor: Windows dispatch (mocked platform)
# ---------------------------------------------------------------------------


def test_stop_supervisor_windows_uses_ctrl_break(tmp_path: Path) -> None:
    """When sys.platform=='win32', stop_supervisor sends CTRL_BREAK_EVENT."""
    from zcu_tools.gui.app.main.services import agent_supervisor as mod

    CTRL_BREAK_EVENT = 1  # signal value on Windows

    with (
        patch.object(mod, "sys") as mock_sys,
        patch("os.kill") as mock_kill,
        patch("signal.CTRL_BREAK_EVENT", CTRL_BREAK_EVENT, create=True),
    ):
        mock_sys.platform = "win32"
        mod.stop_supervisor(12345)

    mock_kill.assert_called_once_with(12345, CTRL_BREAK_EVENT)


def test_stop_supervisor_windows_fallback_to_taskkill(tmp_path: Path) -> None:
    """If CTRL_BREAK_EVENT raises, taskkill must be attempted."""
    from zcu_tools.gui.app.main.services import agent_supervisor as mod

    CTRL_BREAK_EVENT = 1

    with (
        patch.object(mod, "sys") as mock_sys,
        patch("os.kill", side_effect=OSError("access denied")),
        patch("signal.CTRL_BREAK_EVENT", CTRL_BREAK_EVENT, create=True),
        patch("subprocess.run") as mock_run,
    ):
        mock_sys.platform = "win32"
        mod.stop_supervisor(12345)

    # taskkill must have been called with the right pid.
    assert mock_run.called
    args, _ = mock_run.call_args
    cmd = args[0]
    assert "taskkill" in cmd
    assert str(12345) in cmd
