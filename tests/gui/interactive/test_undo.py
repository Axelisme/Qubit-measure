"""Single-level undo observed through the shared interactive session seam."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.interactive import Session
from zcu_tools.gui.session.adapters.manual_owner_scheduler import ManualOwnerScheduler


def test_undo_restores_only_last_commit_and_consumes_history() -> None:
    session = Session([1.0], ManualOwnerScheduler())
    assert not session.can_undo()
    session.commit(lambda state: [2.0])
    session.commit(lambda state: [3.0])
    observed: list[list[float]] = []
    session.subscribe(lambda: observed.append(session.snapshot()))

    assert session.can_undo()
    restored = session.undo()
    assert restored == [2.0]
    restored[0] = 99.0
    assert session.snapshot() == [2.0]
    assert observed == [[2.0]]
    assert not session.can_undo()
    with pytest.raises(FailedPreconditionError, match="undo"):
        session.undo()
    assert session.snapshot() == [2.0]
    assert observed == [[2.0]]

    session.commit(lambda state: [4.0])
    assert session.undo() == [2.0]


def test_failed_commit_preserves_previous_history() -> None:
    session = Session([1.0], ManualOwnerScheduler())
    session.commit(lambda state: [2.0])

    def reject(state: list[float]) -> list[float]:
        state[0] = 9.0
        raise ValueError("invalid candidate")

    with pytest.raises(ValueError, match="invalid candidate"):
        session.commit(reject)
    assert session.snapshot() == [2.0]
    assert session.undo() == [1.0]


@pytest.mark.parametrize("copies_remaining", [0, 1])
def test_candidate_or_return_copy_failure_preserves_history(
    copies_remaining: int,
) -> None:
    @dataclass
    class CopyBudget:
        remaining: int

        def __deepcopy__(self, memo: dict[int, object]) -> CopyBudget:
            if self.remaining == 0:
                raise ValueError("copy budget exhausted")
            return CopyBudget(self.remaining - 1)

    session = Session[object](None, ManualOwnerScheduler())
    session.commit(lambda state: 2)
    with pytest.raises(ValueError, match="copy budget exhausted"):
        session.commit(lambda state: CopyBudget(copies_remaining))
    assert session.snapshot() == 2
    assert session.undo() is None


def test_none_is_a_valid_previous_state() -> None:
    session = Session[int | None](None, ManualOwnerScheduler())
    session.commit(lambda state: 1)
    assert session.can_undo()
    assert session.undo() is None
    assert session.snapshot() is None
    assert not session.can_undo()


def test_undo_subscriber_failure_does_not_revert_or_block_delivery(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = Session(1, ManualOwnerScheduler())
    session.commit(lambda state: 2)
    observed: list[int] = []

    def fail() -> None:
        raise RuntimeError("broken undo view")

    session.subscribe(fail)
    session.subscribe(lambda: observed.append(session.snapshot()))
    with caplog.at_level(logging.ERROR):
        assert session.undo() == 1
    assert observed == [1]
    assert session.snapshot() == 1
    assert not session.can_undo()
    assert "broken undo view" in caplog.text


def test_notification_rejects_reentrant_commit_and_undo() -> None:
    session = Session(1, ManualOwnerScheduler())

    completed: list[int] = []

    def check_reentrancy() -> None:
        with pytest.raises(RuntimeError, match="notification"):
            session.commit(lambda state: 9)
        with pytest.raises(RuntimeError, match="notification"):
            session.undo()
        completed.append(session.snapshot())

    session.subscribe(check_reentrancy)
    session.commit(lambda state: 2)
    assert session.snapshot() == 2
    assert session.undo() == 1
    assert completed == [2, 1]
    assert not session.can_undo()


def test_terminal_input_and_disposal_reject_undo() -> None:
    session = Session(1, ManualOwnerScheduler())
    session.commit(lambda state: 2)
    session.close_input()
    assert session.snapshot() == 2
    assert not session.can_undo()
    with pytest.raises(FailedPreconditionError, match="closed"):
        session.undo()
    session.dispose()
    with pytest.raises(FailedPreconditionError, match="disposed"):
        session.can_undo()
    with pytest.raises(FailedPreconditionError, match="disposed"):
        session.undo()


def test_undo_and_history_queries_require_owner_loop() -> None:
    session = Session(1, ManualOwnerScheduler())
    session.commit(lambda state: 2)
    with ThreadPoolExecutor(max_workers=1) as worker:
        for operation in (session.can_undo, session.undo):
            future = worker.submit(operation)
            with pytest.raises(RuntimeError, match="owner loop"):
                future.result()
    assert session.snapshot() == 2
    assert session.undo() == 1
