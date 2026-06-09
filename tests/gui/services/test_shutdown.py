"""Qt-free ShutdownCoordinator: begin (cancel-all) + tick (poll) state machine.

No Qt loop and a fake clock — the coordinator only decides *when* to close. It
drives OperationHandles (cancel_all / poll), not the exclusion gate (ADR-0019)."""

from __future__ import annotations

import threading

import pytest
from zcu_tools.gui.app.main.services.operation_handles import (
    OperationHandles,
    OperationOutcome,
)
from zcu_tools.gui.app.main.services.shutdown import ShutdownCoordinator, ShutdownState


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t


def _coordinator(handles: OperationHandles, clock: _FakeClock, timeout: float = 10.0):
    return ShutdownCoordinator(handles, timeout=timeout, now=clock.now)


def test_settles_immediately_when_no_operations_active() -> None:
    handles = OperationHandles()
    clock = _FakeClock()
    coord = _coordinator(handles, clock)

    coord.begin()
    assert coord.tick() is ShutdownState.SETTLED
    assert not coord.is_active


def test_waits_until_a_cancellable_operation_self_settles() -> None:
    handles = OperationHandles()
    clock = _FakeClock()
    stop_event = threading.Event()
    token = handles.create(stop_event=stop_event)
    coord = _coordinator(handles, clock)

    coord.begin()
    assert stop_event.is_set()  # begin cancelled it
    assert coord.tick() is ShutdownState.WAITING  # worker has not settled yet

    # Worker self-judged cancelled and settled the handle.
    handles.settle(token, OperationOutcome("cancelled"))
    assert coord.tick() is ShutdownState.SETTLED


def test_times_out_when_an_operation_never_settles() -> None:
    # A connect has no stop_event: cancel is a no-op, so it never settles and
    # only the deadline ends the wait.
    handles = OperationHandles()
    clock = _FakeClock()
    handles.create()  # no stop_event
    coord = _coordinator(handles, clock, timeout=5.0)

    coord.begin()
    assert coord.tick() is ShutdownState.WAITING

    clock.t = 5.0  # deadline reached
    assert coord.tick() is ShutdownState.TIMED_OUT
    assert not coord.is_active


def test_begin_is_idempotent_while_active() -> None:
    handles = OperationHandles()
    clock = _FakeClock()
    handles.create(stop_event=threading.Event())
    coord = _coordinator(handles, clock)

    coord.begin()
    coord.begin()  # second begin while active: no-op, must not re-cancel/extend
    assert coord.is_active


def test_tick_before_begin_raises() -> None:
    handles = OperationHandles()
    clock = _FakeClock()
    coord = _coordinator(handles, clock)
    with pytest.raises(RuntimeError, match="before begin"):
        coord.tick()
