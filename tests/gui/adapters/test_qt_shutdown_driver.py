"""QtShutdownDriver: the QTimer-driven adapter around ShutdownCoordinator.

Verifies the two ends — immediate settle (no operations) and the polled wait
until an operation releases — without reaching into the coordinator."""

from __future__ import annotations

import threading
import time

from qtpy.QtCore import QCoreApplication
from zcu_tools.gui.adapters.qt_shutdown_driver import QtShutdownDriver
from zcu_tools.gui.services.operation_gate import (
    OperationGate,
    OperationKind,
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
    gate = OperationGate()
    driver = QtShutdownDriver(gate)
    closed: list[bool] = []

    driver.begin(lambda: closed.append(True))

    # No active operation: the first synchronous tick settles and on_closed runs
    # without an event-loop round-trip.
    assert closed == [True]


def test_begin_waits_then_closes_when_operation_releases(qapp) -> None:
    del qapp
    gate = OperationGate()
    stop_event = threading.Event()
    lease = gate.acquire(
        OperationKind.DEVICE_SETUP, owner_id="d", stop_event=stop_event
    )
    driver = QtShutdownDriver(gate)
    closed: list[bool] = []

    driver.begin(lambda: closed.append(True))
    assert stop_event.is_set()  # cancelled
    assert closed == []  # still waiting — lease not released

    gate.release(lease, OperationOutcome("cancelled"))
    assert _spin(lambda: closed == [True]), "driver did not close after release"


def test_timeout_forces_close(qapp) -> None:
    del qapp
    gate = OperationGate()
    # A connect has no stop_event → never settles → only the timeout closes it.
    gate.acquire(OperationKind.SOC_CONNECT, owner_id="soc")
    driver = QtShutdownDriver(gate, timeout=0.05)
    closed: list[bool] = []

    driver.begin(lambda: closed.append(True))
    assert closed == []  # waiting on the unstoppable connect

    assert _spin(lambda: closed == [True]), "driver did not force-close on timeout"
