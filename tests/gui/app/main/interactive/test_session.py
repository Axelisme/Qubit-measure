"""Public committed-state contract for interactive analysis."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest
from zcu_tools.gui.app.main.interactive import Session
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


@dataclass
class Pick:
    positions: list[float]
    conjugate: bool = False


def test_complete_commit_notifies_detached_snapshots() -> None:
    source = Pick([1.0, 2.0])
    session = Session(source, ManualOwnerScheduler())
    source.positions[0] = 99.0
    first = session.snapshot()
    assert first.positions == [1.0, 2.0]
    first.positions[0] = 88.0
    observed: list[Pick] = []
    unsubscribe = session.subscribe(lambda: observed.append(session.snapshot()))

    candidate = Pick([3.0, 4.0], True)
    result = session.commit(lambda current: candidate)
    candidate.positions[0] = 77.0
    result.positions[1] = 66.0
    assert session.snapshot() == Pick([3.0, 4.0], True)
    assert observed == [Pick([3.0, 4.0], True)]
    unsubscribe()
    unsubscribe()
    session.commit(lambda current: Pick([5.0, 6.0]))
    assert observed == [Pick([3.0, 4.0], True)]


def test_failed_update_keeps_previous_state_and_isolation() -> None:
    left = Session(Pick([1.0, 2.0]), ManualOwnerScheduler())
    right = Session(Pick([9.0, 8.0]), ManualOwnerScheduler())
    observed: list[Pick] = []
    left.subscribe(lambda: observed.append(left.snapshot()))

    def reject(candidate: Pick) -> Pick:
        candidate.positions[0] = 42.0
        raise ValueError("invalid position")

    with pytest.raises(ValueError, match="invalid position"):
        left.commit(reject)
    assert left.snapshot() == Pick([1.0, 2.0])
    assert right.snapshot() == Pick([9.0, 8.0])
    assert observed == []


def test_notification_failure_does_not_undo_commit_or_block_other_listeners(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = Session(Pick([1.0, 2.0]), ManualOwnerScheduler())
    observed: list[Pick] = []

    def fail() -> None:
        raise RuntimeError("broken view")

    session.subscribe(fail)
    session.subscribe(lambda: observed.append(session.snapshot()))
    with caplog.at_level(logging.ERROR):
        session.commit(lambda current: Pick([2.0, 3.0]))
    assert session.snapshot() == Pick([2.0, 3.0])
    assert observed == [Pick([2.0, 3.0])]
    assert "broken view" in caplog.text


def test_mutations_require_owner_loop_and_leave_other_sessions_unchanged() -> None:
    owner = ManualOwnerScheduler()
    session = Session(Pick([1.0, 2.0]), owner)
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(session.commit, lambda current: Pick([3.0, 4.0]))
        with pytest.raises(RuntimeError, match="owner loop"):
            future.result()
    assert session.snapshot() == Pick([1.0, 2.0])


def test_terminal_and_dispose_close_input_without_late_callbacks() -> None:
    session = Session(Pick([1.0, 2.0]), ManualOwnerScheduler())
    observed: list[Pick] = []
    unsubscribe = session.subscribe(lambda: observed.append(session.snapshot()))
    session.close_input()
    with pytest.raises(FailedPreconditionError):
        session.commit(lambda current: Pick([3.0, 4.0]))
    assert session.snapshot() == Pick([1.0, 2.0])
    session.dispose()
    unsubscribe()
    with pytest.raises(FailedPreconditionError):
        session.snapshot()
    with pytest.raises(FailedPreconditionError):
        session.subscribe(lambda: None)
    assert observed == []
