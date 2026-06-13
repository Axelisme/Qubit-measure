"""Tests for IndependentAgentSession (B1b-1).

No real supervisor or claude process is spawned.  Tests exercise:
  I1  - start() creates session_dir and calls spawn_supervisor_detached.
  I2  - start() begins QTimer poll-tail (timer.isActive).
  I3  - start() while supervisor alive is ignored (fast-fail).
  I4  - poll-tail (tick) reads fixture log and routes updates through callbacks.
  I5  - poll-tail increments byte offset on each tick (incremental reads).
  I6  - send_user_message writes a spool file with correct content.
  I7  - send_user_message is no-op when no supervisor is running.
  I8  - stop() calls stop_supervisor with the correct pid.
  I9  - stop() transitions state to stopped.
  I10 - state transitions: idle→working on start, working→idle on result frame.
  I11 - add_state_listener receives transitions; exceptions are swallowed.
  I12 - session_id populated from system/init frame in log.
  I13 - supervisor disappearance without result frame forces stopped state.
  I14 - platform stop dispatch mock: POSIX vs Windows (via os.kill mock).
  I15 - _read_log_tail raises OSError for missing file (log not yet created).
  I16 - IndependentAgentSession satisfies AgentSessionPort at runtime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.services.agent_supervisor import SupervisorHandle

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_session(
    *,
    has_pending: bool = False,
    parent: object = None,
):
    """Build an IndependentAgentSession with stub callbacks, no real Qt parent."""
    from zcu_tools.gui.app.main.services.independent_agent_session import (
        IndependentAgentSession,
    )

    updates_received: list = []
    states_received: list = []
    errors_received: list = []

    session = IndependentAgentSession(
        on_update=updates_received.append,
        on_state_changed=states_received.append,
        on_process_error=errors_received.append,
        has_pending_wait=lambda: has_pending,
        parent=None,
    )
    return session, updates_received, states_received, errors_received


def _fake_handle(session_dir: Path) -> SupervisorHandle:
    from zcu_tools.gui.app.main.services.agent_supervisor import SupervisorHandle

    return SupervisorHandle(
        pid=42,
        log_path=session_dir / "log.ndjson",
        spool_dir=session_dir / "spool",
    )


def _write_log_lines(log_path: Path, lines: list[str]) -> None:
    """Append NDJSON lines to a log fixture file."""
    with log_path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.rstrip("\n") + "\n")


def _system_init_line(session_id: str = "sid-test") -> str:
    return json.dumps({"type": "system", "subtype": "init", "session_id": session_id})


def _assistant_line(text: str = "Hello") -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        }
    )


def _result_line(is_error: bool = False) -> str:
    return json.dumps(
        {
            "type": "result",
            "is_error": is_error,
            "result": "done",
            "total_cost_usd": 0.0,
            "terminal_reason": "completed" if not is_error else "error",
        }
    )


# ---------------------------------------------------------------------------
# I1 — start() creates session_dir and calls spawn_supervisor_detached
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_start_calls_spawn_supervisor(tmp_path: Path) -> None:
    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session.spawn_supervisor_detached",
        return_value=fake_handle,
    ) as mock_spawn:
        session.start("do thing", "/repo")

    assert mock_spawn.call_count == 1
    _, pos_args, kw_args = mock_spawn.mock_calls[0]
    # First positional arg is session_dir (a Path), second is task, third is repo_root.
    _session_dir, task, repo_root = pos_args
    assert task == "do thing"
    assert repo_root == "/repo"


# ---------------------------------------------------------------------------
# I2 — start() begins QTimer
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_start_activates_timer(tmp_path: Path) -> None:
    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session.spawn_supervisor_detached",
        return_value=fake_handle,
    ):
        session.start("task", "/repo")

    assert session._timer.isActive()
    session._timer.stop()  # clean up


# ---------------------------------------------------------------------------
# I3 — start() while supervisor alive is ignored
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_start_ignored_when_supervisor_alive(tmp_path: Path) -> None:
    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    session._handle = fake_handle

    with (
        patch(
            "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
            return_value=True,
        ),
        patch(
            "zcu_tools.gui.app.main.services.independent_agent_session.spawn_supervisor_detached"
        ) as mock_spawn,
    ):
        session.start("task", "/repo")

    mock_spawn.assert_not_called()


# ---------------------------------------------------------------------------
# I4 — poll-tail (tick) reads fixture log and routes updates
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_tick_reads_log_and_routes_updates(tmp_path: Path) -> None:
    session, updates, states, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)
    session._handle = fake_handle
    session._run_state.on_start()

    _write_log_lines(fake_handle.log_path, [_assistant_line("Hi!")])

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=True,
    ):
        session.tick()

    from zcu_tools.gui.app.main.services.agent_runner import AssistantTextUpdate

    assistant_updates = [
        u for batch in updates for u in batch if isinstance(u, AssistantTextUpdate)
    ]
    assert len(assistant_updates) == 1
    assert assistant_updates[0].text == "Hi!"


# ---------------------------------------------------------------------------
# I5 — poll-tail increments byte offset (incremental reads)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_tick_increments_offset(tmp_path: Path) -> None:
    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)
    session._handle = fake_handle
    session._run_state.on_start()

    line1 = _assistant_line("first")
    _write_log_lines(fake_handle.log_path, [line1])

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=True,
    ):
        session.tick()

    offset_after_first = session._log_offset
    assert offset_after_first > 0

    # Append a second line and tick again.
    line2 = _assistant_line("second")
    _write_log_lines(fake_handle.log_path, [line2])

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=True,
    ):
        session.tick()

    assert session._log_offset > offset_after_first


# ---------------------------------------------------------------------------
# I6 — send_user_message writes a spool file
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_send_user_message_writes_spool(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.agent_supervisor import consume_spool_entries

    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)
    session._handle = fake_handle
    session._run_state.on_start()

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=True,
    ):
        session.send_user_message("hello from user")

    entries = consume_spool_entries(fake_handle.spool_dir)
    assert len(entries) == 1
    obj = json.loads(entries[0].read_bytes())
    assert obj["message"]["content"] == "hello from user"


# ---------------------------------------------------------------------------
# I7 — send_user_message is no-op when no supervisor
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_send_user_message_noop_without_supervisor() -> None:

    session, _, _, _ = _make_session()
    # No handle set — must not raise.
    session.send_user_message("ignored")  # should log warning and return


# ---------------------------------------------------------------------------
# I8 — stop() calls stop_supervisor with correct pid
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_stop_calls_stop_supervisor(tmp_path: Path) -> None:
    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    session._handle = fake_handle
    session._run_state.on_start()

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session.stop_supervisor"
    ) as mock_stop:
        session.stop()

    mock_stop.assert_called_once_with(fake_handle.pid)


# ---------------------------------------------------------------------------
# I9 — stop() transitions state to stopped
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_stop_transitions_to_stopped(tmp_path: Path) -> None:
    session, _, states, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    session._handle = fake_handle
    session._run_state.on_start()

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session.stop_supervisor"
    ):
        session.stop()

    assert session.state == "stopped"
    assert "stopped" in states


# ---------------------------------------------------------------------------
# I10 — state transitions: idle → working → idle
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_state_transitions_via_log(tmp_path: Path) -> None:
    session, updates, states, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)
    session._handle = fake_handle

    assert session.state == "idle"
    session._run_state.on_start()
    assert session.state == "working"

    # Feed a result frame via the log.
    _write_log_lines(fake_handle.log_path, [_result_line(is_error=False)])

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=True,
    ):
        session.tick()

    assert session.state == "idle"


# ---------------------------------------------------------------------------
# I11 — add_state_listener: receives transitions, exceptions swallowed
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_add_state_listener_receives_changes(tmp_path: Path) -> None:
    session, _, _, _ = _make_session()
    received: list = []
    session.add_state_listener(received.append)

    session._run_state.on_start()
    session._emit_state()

    assert "working" in received


@pytest.mark.usefixtures("qapp")
def test_add_state_listener_exception_swallowed() -> None:
    session, _, _, _ = _make_session()
    second_calls: list = []

    def boom(s: object) -> None:
        raise RuntimeError("boom")

    session.add_state_listener(boom)
    session.add_state_listener(second_calls.append)

    session._run_state.on_start()
    session._emit_state()  # must not raise

    assert "working" in second_calls


@pytest.mark.usefixtures("qapp")
def test_add_state_listener_duplicate_ignored() -> None:
    session, _, _, _ = _make_session()
    calls: list = []
    cb = calls.append
    session.add_state_listener(cb)
    session.add_state_listener(cb)  # duplicate

    session._run_state.on_start()
    session._emit_state()

    assert len(calls) == 1  # fired once, not twice


# ---------------------------------------------------------------------------
# I12 — session_id populated from system/init frame
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_session_id_populated_from_log(tmp_path: Path) -> None:
    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)
    session._handle = fake_handle
    session._run_state.on_start()

    _write_log_lines(fake_handle.log_path, [_system_init_line("my-session-123")])

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=True,
    ):
        session.tick()

    assert session.session_id() == "my-session-123"


# ---------------------------------------------------------------------------
# I13 — supervisor disappearance without result frame → forced stopped
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_supervisor_disappearance_forces_stopped(tmp_path: Path) -> None:
    session, _, states, errors = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.log_path.touch()  # log exists but empty
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)
    session._handle = fake_handle
    session._run_state.on_start()

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=False,
    ):
        session.tick()

    assert session.state == "stopped"
    assert len(errors) == 1
    assert "disappeared" in errors[0]


# ---------------------------------------------------------------------------
# I14 — platform stop dispatch (os.kill mock)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
def test_stop_sends_sigint_on_posix(tmp_path: Path) -> None:

    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    session._handle = fake_handle
    session._run_state.on_start()

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session.stop_supervisor"
    ) as mock_stop:
        session.stop()

    mock_stop.assert_called_once_with(fake_handle.pid)


# ---------------------------------------------------------------------------
# I15 — _read_log_tail raises OSError for missing file
# ---------------------------------------------------------------------------


def test_read_log_tail_raises_for_missing_file(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.independent_agent_session import _read_log_tail

    missing = tmp_path / "nonexistent.ndjson"
    with pytest.raises(OSError):
        _read_log_tail(missing, 0, 4096)


def test_read_log_tail_reads_from_offset(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.independent_agent_session import _read_log_tail

    log_path = tmp_path / "log.ndjson"
    log_path.write_bytes(b"line1\nline2\n")
    # Skip the first 6 bytes ("line1\n").
    data = _read_log_tail(log_path, 6, 4096)
    assert data == b"line2\n"


def test_read_log_tail_returns_empty_at_end(tmp_path: Path) -> None:
    from zcu_tools.gui.app.main.services.independent_agent_session import _read_log_tail

    log_path = tmp_path / "log.ndjson"
    log_path.write_bytes(b"line1\n")
    data = _read_log_tail(log_path, 6, 4096)
    assert data == b""


# ---------------------------------------------------------------------------
# I16 — IndependentAgentSession satisfies AgentSessionPort at runtime
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_independent_agent_session_conforms_to_port() -> None:
    from zcu_tools.gui.app.main.services.ports import AgentSessionPort

    session, _, _, _ = _make_session()
    assert isinstance(session, AgentSessionPort)


# ---------------------------------------------------------------------------
# I17 — start() uses registry_dir (B1b-2: named dir, not tmpdir)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_start_uses_registry_dir(tmp_path: Path) -> None:
    """start() must use registry_dir()/<session_id>/ not a tmpdir."""
    from unittest.mock import patch

    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)

    captured_dirs: list[Path] = []

    def _fake_spawn(session_dir, task, repo_root, *, session_id=None):
        captured_dirs.append(Path(session_dir))
        return fake_handle

    with (
        patch(
            "zcu_tools.gui.app.main.services.independent_agent_session.registry_dir",
            return_value=tmp_path,
        ),
        patch(
            "zcu_tools.gui.app.main.services.independent_agent_session.spawn_supervisor_detached",
            side_effect=_fake_spawn,
        ),
    ):
        session.start("my task", "/repo")

    assert len(captured_dirs) == 1
    # The session_dir must be a subdirectory of registry_dir (tmp_path).
    assert captured_dirs[0].parent == tmp_path
    session._timer.stop()


# ---------------------------------------------------------------------------
# I18 — attach() rebuilds transcript from existing log fixture
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_attach_replays_log(tmp_path: Path) -> None:
    """attach() must reset offset=0 and replay the full log through callbacks."""
    from unittest.mock import patch

    from zcu_tools.gui.app.main.services.agent_runner import AssistantTextUpdate

    session, updates, states, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)

    # Pre-write log: system init + assistant text + result.
    _write_log_lines(
        fake_handle.log_path,
        [
            _system_init_line("sid-attach"),
            _assistant_line("Hello from attach"),
            _result_line(is_error=False),
        ],
    )

    record = {
        "session_id": "test0001",
        "claude_session_id": "",
        "pid": fake_handle.pid,
        "status": "stopped",
        "log_path": str(fake_handle.log_path),
        "spool_dir": str(fake_handle.spool_dir),
        "created": "2026-06-14T00:00:00",
        "title": "test attach",
    }

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=False,  # stopped session
    ):
        session.attach(record)  # type: ignore[arg-type]
        session.tick()

    assistant_updates = [
        u for batch in updates for u in batch if isinstance(u, AssistantTextUpdate)
    ]
    assert len(assistant_updates) >= 1
    assert any(u.text == "Hello from attach" for u in assistant_updates)
    assert session.session_id() == "sid-attach"
    session._timer.stop()


# ---------------------------------------------------------------------------
# I19 — attach() stopped record: state eventually transitions to stopped
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_attach_stopped_record_transitions_stopped(tmp_path: Path) -> None:
    """attach() of a stopped session: after reading the result frame, state=stopped."""
    from unittest.mock import patch

    session, _, states, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)

    _write_log_lines(
        fake_handle.log_path,
        [_result_line(is_error=False)],
    )

    record = {
        "session_id": "test0002",
        "claude_session_id": "",
        "pid": fake_handle.pid,
        "status": "stopped",
        "log_path": str(fake_handle.log_path),
        "spool_dir": str(fake_handle.spool_dir),
        "created": "2026-06-14T00:00:00",
        "title": "test stopped attach",
    }

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=False,
    ):
        session.attach(record)  # type: ignore[arg-type]
        session.tick()

    assert session.state == "idle"
    session._timer.stop()


# ---------------------------------------------------------------------------
# I20 — detach() stops timer without calling stop_supervisor
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_detach_stops_timer_not_supervisor(tmp_path: Path) -> None:
    """detach() must stop the timer and clear handle; stop_supervisor must NOT be called."""
    from unittest.mock import patch

    session, _, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    session._handle = fake_handle
    session._run_state.on_start()
    session._timer.start()

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session.stop_supervisor"
    ) as mock_stop:
        session.detach()

    mock_stop.assert_not_called()
    assert session._handle is None
    assert not session._timer.isActive()


# ---------------------------------------------------------------------------
# I21 — detach() then attach() re-starts the tail from offset=0
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("qapp")
def test_detach_then_attach_replays(tmp_path: Path) -> None:
    """After detach(), a subsequent attach() must replay the log from scratch."""
    from unittest.mock import patch

    from zcu_tools.gui.app.main.services.agent_runner import AssistantTextUpdate

    session, updates, _, _ = _make_session()
    fake_handle = _fake_handle(tmp_path)
    fake_handle.spool_dir.mkdir(parents=True, exist_ok=True)

    _write_log_lines(fake_handle.log_path, [_assistant_line("Re-attached")])

    session._handle = fake_handle
    session._run_state.on_start()
    session._log_offset = 999  # simulate an earlier offset
    session.detach()

    record = {
        "session_id": "test0003",
        "claude_session_id": "",
        "pid": fake_handle.pid,
        "status": "stopped",
        "log_path": str(fake_handle.log_path),
        "spool_dir": str(fake_handle.spool_dir),
        "created": "2026-06-14T00:00:00",
        "title": "re-attach",
    }

    with patch(
        "zcu_tools.gui.app.main.services.independent_agent_session._supervisor_alive",
        return_value=False,
    ):
        session.attach(record)  # type: ignore[arg-type]
        assert session._log_offset == 0
        session.tick()

    all_updates = [
        u for batch in updates for u in batch if isinstance(u, AssistantTextUpdate)
    ]
    assert any(u.text == "Re-attached" for u in all_updates)
    session._timer.stop()
