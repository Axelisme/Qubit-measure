"""QtShutdownDriver: the QTimer-driven adapter around ShutdownCoordinator.

Verifies the two ends — immediate settle (no operations) and the polled wait
until an operation settles — without reaching into the coordinator. Operations
are modelled directly as OperationHandles tokens (ADR-0019)."""

from __future__ import annotations

import time

from qtpy.QtCore import QCoreApplication
from zcu_tools.gui.app.main.driven.qt_shutdown_driver import QtShutdownDriver
from zcu_tools.gui.session.operation_handles import (
    OperationHandles,
    OperationOutcome,
)


def _spin(condition, timeout_ms: int = 3000, step_ms: int = 10) -> bool:
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(step_ms / 1000)
    return False


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
    token = handles.create(cancel_hook=lambda: hook_called.append(True))
    driver = QtShutdownDriver(handles)
    closed: list[bool] = []

    driver.begin(lambda: closed.append(True))
    assert hook_called == [True]  # cancel_hook invoked on cancel_all
    assert closed == []  # still waiting — handle not settled

    handles.settle(token, OperationOutcome("cancelled"))
    assert _spin(lambda: closed == [True]), "driver did not close after settle"


def test_timeout_forces_close(qapp) -> None:
    del qapp
    handles = OperationHandles()
    # A connect has no stop_event → never settles → only the timeout closes it.
    handles.create()
    driver = QtShutdownDriver(handles, timeout=0.05)
    closed: list[bool] = []

    driver.begin(lambda: closed.append(True))
    assert closed == []  # waiting on the unstoppable connect

    assert _spin(lambda: closed == [True]), "driver did not force-close on timeout"
