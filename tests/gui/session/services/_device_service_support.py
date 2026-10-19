"""Shared DeviceService composition and connection helpers for service tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationInfrastructure
from zcu_tools.gui.session.ports import BackgroundExecutor
from zcu_tools.gui.session.services.device import ConnectDeviceRequest, DeviceService
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui._completion_helpers import (
    on_device_connected,
    on_device_operation_failed,
)
from tests.gui._qt_waits import qt_until
from tests.gui.session.services._device_fakes import FakeDeviceRegistry
from tests.gui.session.services._device_waits import DeviceWorkers


def make_device_service(
    workers: DeviceWorkers,
    gate: OperationGate | None = None,
    state: State | None = None,
    bus: EventBus | None = None,
    handles: OperationHandles | None = None,
    bg: BackgroundExecutor | None = None,
) -> tuple[DeviceService, MagicMock]:
    """Build a service and mock driver using this test's worker owner.

    Optional gate, state, bus, handles and background executor override the
    existing defaults. Workers retain default runners and observe bus failures;
    the caller must close the same owner after releasing test-specific holds.
    """
    from zcu_tools.gui.session.operation_runner import OperationRunner

    device = MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    bg = bg or workers.background()
    resolved_gate = gate or OperationGate(EventBus())
    handles = handles or OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = bus or EventBus()
    workers.watch(bus)
    runner = OperationRunner(resolved_gate, handles, progress, bg, bus)
    return (
        DeviceService(
            bus,
            state or State(MagicMock()),
            OperationInfrastructure(
                gate=resolved_gate,
                background=bg,
                runner=runner,
                handles=handles,
            ),
            driver_factory=lambda type_name, address: device,
            device_registry=FakeDeviceRegistry(),
        ),
        device,
    )


def device_connect_request(name: str = "dev1") -> ConnectDeviceRequest:
    """Request a FakeDevice named ``name`` at the mock driver address."""
    return ConnectDeviceRequest(type_name="FakeDevice", name=name, address="addr")


def connect_device(svc: DeviceService, req: ConnectDeviceRequest) -> None:
    """Start ``req`` and dispatch Qt until success or failure; assert no failure.

    Call on the Qt owner thread. The shared Qt watchdog raises on timeout.
    """
    connected: list[object] = []
    errors: list[str] = []
    on_device_connected(svc, connected.append)
    on_device_operation_failed(svc, lambda _name, error: errors.append(error))
    svc.start_connect_device(req)
    qt_until(lambda: bool(connected or errors), label=f"connect {req.name}")
    assert not errors
