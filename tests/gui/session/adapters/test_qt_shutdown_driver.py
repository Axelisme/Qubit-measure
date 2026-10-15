"""QtShutdownDriver: the QTimer-driven adapter around ShutdownCoordinator.

Verifies the two ends — immediate settle (no operations) and the polled wait
until an operation settles — without reaching into the coordinator. Operations
are modelled directly as OperationHandles tokens (ADR-0066)."""

from __future__ import annotations

import time

from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.session.adapters.qt_shutdown_driver import QtShutdownDriver
from zcu_tools.gui.session.operation_handles import (
    OperationHandles,
    OperationOutcome,
)

from tests.gui._lifecycle_waits import running
from tests.gui._qt_waits import qt_until as _spin


def test_begin_closes_immediately_when_idle(qapp) -> None:
    del qapp
    handles = OperationHandles()
    driver = QtShutdownDriver(handles)
    closed: list[bool] = []

    driver.begin(lambda: closed.append(True))

    # No active operation: the first synchronous tick settles and on_closed runs
    # without an event-loop round-trip.
    assert closed == [True]


def test_begin_waits_then_closes_when_operation_settles(qapp) -> None:
    del qapp
    handles = OperationHandles()
    hook_called: list[bool] = []
    token = handles.create(
        cancel_hook=lambda: hook_called.append(True), origin=EventOrigin(kind="user")
    )
    driver = QtShutdownDriver(handles)
    closed: list[bool] = []

    driver.begin(lambda: closed.append(True))
    assert hook_called == [True]  # cancel_hook invoked on cancel_all
    assert closed == []  # still waiting — handle not settled

    with running(lambda: handles.settle(token, OperationOutcome("cancelled"))):
        _spin(lambda: closed == [True], label="driver did not close after settle")


def test_timeout_forces_close(qapp, monkeypatch) -> None:
    del qapp
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    handles = OperationHandles()
    # A connect has no stop_event → never settles → only the timeout closes it.
    handles.create(origin=EventOrigin(kind="user"))
    driver = QtShutdownDriver(handles, timeout=10)
    closed: list[bool] = []

    driver.begin(lambda: closed.append(True))
    assert closed == []  # waiting on the unstoppable connect

    now[0] += 10
    _spin(lambda: closed == [True], label="driver did not force-close on timeout")


def test_close_callback_exception_is_logged(qapp, caplog) -> None:
    del qapp
    handles = OperationHandles()
    driver = QtShutdownDriver(handles)

    def close_boom() -> None:
        raise RuntimeError("close boom")

    with caplog.at_level("ERROR"):
        driver.begin(close_boom)

    assert "shutdown close callback failed" in caplog.text
