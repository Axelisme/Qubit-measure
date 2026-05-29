"""Tests for async DeviceService commands and snapshot boundaries."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.event_bus import EventBus, GuiEvent
from zcu_tools.gui.services.device import (
    ConnectDeviceRequest,
    DeviceService,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetDeviceValueRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.state import State
from zcu_tools.gui.services.operation_gate import (
    OperationConflictError,
    OperationGate,
    OperationKind,
)


@pytest.fixture(autouse=True)
def _clean_devices():
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)
    yield
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)


def _make_svc(
    driver: MagicMock | None = None, gate: OperationGate | None = None
) -> tuple[DeviceService, MagicMock]:
    device = driver or MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=0.0)
    return (
        DeviceService(
            EventBus(),
            State(MagicMock()),
            gate,
            driver_factory=lambda _type, _address: device,
        ),
        device,
    )


def _req(name: str = "dev1") -> ConnectDeviceRequest:
    return ConnectDeviceRequest(type_name="FakeDevice", name=name, address="addr")


def _connect(svc: DeviceService, req: ConnectDeviceRequest) -> None:
    loop = QEventLoop()
    svc.device_connected.connect(lambda _request: loop.quit())
    svc.operation_failed.connect(lambda _name, _error: loop.quit())
    svc.start_connect_device(req)
    loop.exec()


def test_device_connect_disconnect_and_set_value_update_snapshot(qapp):
    svc, device = _make_svc()
    _connect(svc, _req())
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]

    device.get_info.return_value = FakeDeviceInfo(address="addr", value=1.0)
    value_loop = QEventLoop()
    svc.value_set.connect(lambda _name: value_loop.quit())
    svc.start_set_device_value(SetDeviceValueRequest(name="dev1", value=1.0))
    value_loop.exec()
    device.set_value.assert_called_once_with(1.0)
    assert svc.get_device_snapshot("dev1").info.value == 1.0  # type: ignore[union-attr]

    drop_loop = QEventLoop()
    svc.device_disconnected.connect(lambda _request: drop_loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    drop_loop.exec()
    device.close.assert_called_once_with()
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.MEMORY_ONLY  # type: ignore[union-attr]


def test_device_mutation_is_globally_exclusive_and_blocks_same_device_read(qapp):
    gate = OperationGate()
    svc, _device = _make_svc(gate=gate)
    _connect(svc, _req())
    lease = gate.acquire(
        OperationKind.DEVICE_SETUP, owner_id="held", resource_id="dev1"
    )

    with pytest.raises(OperationConflictError, match="Cannot start"):
        svc.start_set_device_value(SetDeviceValueRequest(name="dev1", value=1.0))
    with pytest.raises(OperationConflictError, match="Cannot read"):
        svc.get_device_info("dev1")
    gate.release(lease)


def test_device_connect_failure_is_reported_without_live_registration(qapp):
    svc, device = _make_svc()
    device.get_info.side_effect = RuntimeError("cannot query")
    errors: list[str] = []
    loop = QEventLoop()
    svc.operation_failed.connect(
        lambda _name, error: errors.append(error) or loop.quit()
    )

    svc.start_connect_device(_req())
    loop.exec()

    assert errors
    assert "cannot query" in errors[0]
    assert "dev1" not in GlobalDeviceManager.get_all_devices()
    device.close.assert_called_once_with()
    assert svc.get_device_snapshot("dev1") is None


def test_setup_uses_gate_and_returns_to_connected_snapshot(qapp):
    svc, device = _make_svc()
    _connect(svc, _req())
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=2.0)
    loop = QEventLoop()
    svc.setup_finished.connect(lambda _name: loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=2.0))
    )
    assert svc.get_active_setup() is not None
    loop.exec()

    device.setup.assert_called_once()
    assert svc.get_active_setup() is None
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]


def test_wait_setup_done_wakes_from_worker_thread_without_deadlock(qapp):
    """wait_setup_done runs off the main thread; with the main event loop
    spinning, the setup terminal wakes it promptly (regression: previously it
    deadlocked because the handler occupied the main thread it needed)."""
    import threading
    import time

    svc, device = _make_svc()
    _connect(svc, _req())

    # Make setup take ~0.5s so the waiter genuinely has to block.
    def slow_setup(_info, stop_event=None):
        time.sleep(0.5)

    device.setup.side_effect = slow_setup
    device.get_info.return_value = FakeDeviceInfo(address="addr", value=2.0)

    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=2.0))
    )

    result: dict[str, object] = {}

    def waiter() -> None:
        t0 = time.monotonic()
        try:
            svc.wait_setup_done("dev1", timeout=3.0)
            result["error"] = None
        except RuntimeError as exc:
            result["error"] = str(exc)
        result["dt"] = time.monotonic() - t0

    wt = threading.Thread(target=waiter)
    wt.start()

    # Spin the main event loop (as the real GUI does) so the worker's
    # setup_finished signal is dispatched and the gate lease is released.
    deadline = time.monotonic() + 3.0
    while wt.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    wt.join(timeout=1.0)

    assert result.get("error") is None
    dt = result.get("dt")
    assert isinstance(dt, float)
    assert dt < 2.0  # woke shortly after the 0.5s setup, not on timeout


def test_wait_setup_done_raises_on_setup_failure(qapp):
    import threading
    import time

    svc, device = _make_svc()
    _connect(svc, _req())

    def failing_setup(_info, stop_event=None):
        time.sleep(0.2)
        raise RuntimeError("hardware boom")

    device.setup.side_effect = failing_setup

    svc.start_setup_device(
        SetupDeviceRequest(name="dev1", info=FakeDeviceInfo(address="addr", value=2.0))
    )

    result: dict[str, object] = {}

    def waiter() -> None:
        try:
            svc.wait_setup_done("dev1", timeout=3.0)
            result["error"] = None
        except RuntimeError as exc:
            result["error"] = str(exc)

    wt = threading.Thread(target=waiter)
    wt.start()
    deadline = time.monotonic() + 3.0
    while wt.is_alive() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    wt.join(timeout=1.0)

    assert result.get("error") is not None
    assert "hardware boom" in str(result["error"])


def test_disconnect_close_failure_retains_connected_device_and_releases_gate(qapp):
    svc, device = _make_svc()
    _connect(svc, _req())
    device.close.side_effect = OSError("close failed")
    loop = QEventLoop()
    errors: list[str] = []
    svc.operation_failed.connect(
        lambda _name, error: errors.append(error) or loop.quit()
    )

    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    loop.exec()

    assert "close failed" in errors[0]
    assert "dev1" in GlobalDeviceManager.get_all_devices()
    assert svc.get_device_snapshot("dev1").status is DeviceStatus.CONNECTED  # type: ignore[union-attr]
    device.close.side_effect = None
    retry_loop = QEventLoop()
    svc.device_disconnected.connect(lambda _request: retry_loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    retry_loop.exec()


def test_event_failure_before_worker_start_does_not_leak_device_lease(qapp):
    bus = EventBus()
    svc = DeviceService(
        bus,
        State(MagicMock()),
        driver_factory=lambda _type, _address: MagicMock(),
    )
    bus.subscribe(
        GuiEvent.DEVICE_CHANGED,
        MagicMock(side_effect=RuntimeError("render failed")),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        svc.start_connect_device(_req())

    assert svc.get_device_snapshot("dev1") is None
