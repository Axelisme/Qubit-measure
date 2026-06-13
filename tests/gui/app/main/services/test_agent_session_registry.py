"""Tests for agent_session_registry (B1b-2).

Covers:
  R1  - write_record / read_record round-trip.
  R2  - list_records returns entries sorted by created (oldest first).
  R3  - remove_record is idempotent (no error on double-remove).
  R4  - stale running record: pid dead → read_record returns stopped.
  R5  - atomic tmp file does not linger after write_record.
  R6  - read_record returns None for missing session_id.
  R7  - list_records on empty / missing directory returns empty list.
  R8  - _pid_alive returns False for dead pid (mock, Windows-safe).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(session_id: str, status: str = "running", pid: int = 99999) -> dict:
    return {
        "session_id": session_id,
        "claude_session_id": "",
        "pid": pid,
        "status": status,
        "log_path": "/tmp/fake/log.ndjson",
        "spool_dir": "/tmp/fake/spool",
        "created": f"2026-06-14T00:00:0{session_id[-1]}.000000",
        "title": f"task for {session_id}",
    }


# ---------------------------------------------------------------------------
# R1 — round-trip
# ---------------------------------------------------------------------------


def test_write_read_roundtrip(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_session_registry import (
        read_record,
        write_record,
    )

    rec = _make_record("aabbccdd", status="stopped", pid=12345)

    with patch(
        "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
        return_value=tmp_path,
    ):
        write_record(rec)  # type: ignore[arg-type]
        result = read_record("aabbccdd")

    assert result is not None
    assert result["session_id"] == "aabbccdd"
    assert result["status"] == "stopped"
    assert result["pid"] == 12345
    assert result["title"] == "task for aabbccdd"


# ---------------------------------------------------------------------------
# R2 — list sorted by created
# ---------------------------------------------------------------------------


def test_list_records_sorted_by_created(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_session_registry import (
        list_records,
        write_record,
    )

    recs = [
        _make_record("zzzzzz00", status="stopped", pid=1),
        _make_record("zzzzzz01", status="stopped", pid=2),
        _make_record("zzzzzz02", status="stopped", pid=3),
    ]
    # Intentionally write in reverse order.
    for rec in reversed(recs):
        with patch(
            "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
            return_value=tmp_path,
        ):
            write_record(rec)  # type: ignore[arg-type]

    with patch(
        "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
        return_value=tmp_path,
    ):
        result = list_records()

    assert [r["session_id"] for r in result] == ["zzzzzz00", "zzzzzz01", "zzzzzz02"]


# ---------------------------------------------------------------------------
# R3 — remove_record idempotent
# ---------------------------------------------------------------------------


def test_remove_record_idempotent(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_session_registry import (
        remove_record,
        write_record,
    )

    rec = _make_record("deadbeef", status="stopped")

    with patch(
        "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
        return_value=tmp_path,
    ):
        write_record(rec)  # type: ignore[arg-type]
        remove_record("deadbeef")
        # Second remove must not raise.
        remove_record("deadbeef")


# ---------------------------------------------------------------------------
# R4 — stale running record: dead pid → read_record returns stopped (decision D)
# ---------------------------------------------------------------------------


def test_stale_running_pid_dead_transitions_to_stopped(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_session_registry import (
        read_record,
        write_record,
    )

    # Write a running record with a pid we will mock as dead.
    rec = _make_record("deadpid0", status="running", pid=99998)

    with patch(
        "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
        return_value=tmp_path,
    ):
        write_record(rec)  # type: ignore[arg-type]

    # Patch _pid_alive to return False (simulates dead pid).
    with (
        patch(
            "zcu_tools.gui.app.main.services.agent_session_registry._pid_alive",
            return_value=False,
        ),
        patch(
            "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
            return_value=tmp_path,
        ),
    ):
        result = read_record("deadpid0")

    assert result is not None
    assert result["status"] == "stopped"

    # The on-disk file should also be updated to stopped.
    disk_path = tmp_path / "deadpid0.json"
    disk_content = json.loads(disk_path.read_text())
    assert disk_content["status"] == "stopped"


# ---------------------------------------------------------------------------
# R5 — atomic: no .tmp file after write_record
# ---------------------------------------------------------------------------


def test_no_tmp_file_after_write(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_session_registry import write_record

    rec = _make_record("cleanwrite", status="stopped")

    with patch(
        "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
        return_value=tmp_path,
    ):
        write_record(rec)  # type: ignore[arg-type]

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Leftover tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# R6 — read_record returns None for missing session
# ---------------------------------------------------------------------------


def test_read_record_missing_returns_none(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_session_registry import read_record

    with patch(
        "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
        return_value=tmp_path,
    ):
        result = read_record("nosuchid")

    assert result is None


# ---------------------------------------------------------------------------
# R7 — list_records on empty/missing dir returns empty
# ---------------------------------------------------------------------------


def test_list_records_empty_dir(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_session_registry import list_records

    with patch(
        "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
        return_value=tmp_path,
    ):
        result = list_records()

    assert result == []


def test_list_records_missing_dir(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_session_registry import list_records

    missing = tmp_path / "nonexistent"
    with patch(
        "zcu_tools.gui.app.main.services.agent_session_registry.registry_dir",
        return_value=missing,
    ):
        result = list_records()

    assert result == []


# ---------------------------------------------------------------------------
# R8 — _pid_alive returns False for dead pid (mock, Windows-safe)
# ---------------------------------------------------------------------------


def test_pid_alive_false_for_dead_process() -> None:
    """_pid_alive_posix must return False when os.kill raises ProcessLookupError."""
    from zcu_tools.gui.app.main.services.agent_session_registry import _pid_alive_posix

    # Patch os.kill so we don't accidentally signal a real process.
    with patch("os.kill", side_effect=ProcessLookupError):
        assert _pid_alive_posix(99999) is False


def test_pid_alive_true_when_process_exists() -> None:
    """_pid_alive_posix must return True when os.kill raises PermissionError
    (process exists but belongs to another user)."""
    from zcu_tools.gui.app.main.services.agent_session_registry import _pid_alive_posix

    with patch("os.kill", side_effect=PermissionError):
        assert _pid_alive_posix(1) is True
