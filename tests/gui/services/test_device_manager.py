"""Tests for the device registry, setup worker lifecycle, and progress events.

Two driver styles coexist deliberately: the setup-worker / progress tests use a
``MagicMock`` driver (they assert on call interactions), while the registry CRUD
tests use a real ``FakeDevice`` (they read/write real values through the
GlobalDeviceManager registry).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.device import FakeDevice, FakeDeviceInfo, GlobalDeviceManager
from zcu_tools.gui.event_bus import DeviceSetupChangedPayload, EventBus, GuiEvent
from zcu_tools.gui.services.device import (
    ConnectDeviceRequest,
    DeviceService,
    DisconnectDeviceRequest,
    SetDeviceValueRequest,
    SetupDeviceRequest,
    _DeviceSetupWorker,
)
from zcu_tools.gui.state import State


@pytest.fixture(autouse=True)
def _clean_devices():
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)
    yield
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)


def _make_svc(driver: MagicMock | None = None) -> tuple[DeviceService, MagicMock]:
    device = driver or MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="none")
    svc = DeviceService(
        EventBus(),
        State(MagicMock()),
        driver_factory=lambda _type, _address: device,  # type: ignore[arg-type]
    )
    return svc, device


def _connect(svc: DeviceService) -> None:
    loop = QEventLoop()
    svc.device_connected.connect(lambda _request: loop.quit())
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name="test_dev", address="none")
    )
    loop.exec()


def test_device_setup_worker_success(qapp):
    dev = MagicMock()
    dev.get_info.return_value = FakeDeviceInfo(address="none")
    worker = _DeviceSetupWorker(dev, "test_dev", FakeDeviceInfo(address="none"), None)
    loop = QEventLoop()
    completed: list[bool] = []
    worker.setup_finished.connect(
        lambda _name, _info: completed.append(not worker.isRunning()) or loop.quit()
    )

    worker.start()
    loop.exec()

    dev.setup.assert_called_once()
    assert completed == [True]


def test_device_setup_worker_failure(qapp):
    dev = MagicMock()
    dev.setup.side_effect = RuntimeError("setup failed")
    worker = _DeviceSetupWorker(dev, "test_dev", FakeDeviceInfo(address="none"), None)
    loop = QEventLoop()
    errors: list[str] = []
    worker.failed.connect(lambda _name, error: errors.append(error) or loop.quit())

    worker.start()
    loop.exec()

    assert errors == ["setup failed"]


def test_device_setup_worker_cancel(qapp):
    dev = MagicMock()
    dev.get_info.return_value = FakeDeviceInfo(address="none")
    dev.setup.side_effect = lambda _info, stop_event: stop_event.wait(0.1)
    worker = _DeviceSetupWorker(dev, "test_dev", FakeDeviceInfo(address="none"), None)
    loop = QEventLoop()
    cancelled: list[str] = []
    worker.cancelled.connect(lambda name: cancelled.append(name) or loop.quit())

    worker.start()
    worker.cancel()
    loop.exec()

    assert cancelled == ["test_dev"]


def test_device_service_emits_active_and_terminal_setup_snapshots(qapp):
    svc, _device = _make_svc()
    _connect(svc)
    events: list[DeviceSetupChangedPayload] = []
    svc._bus.subscribe(GuiEvent.DEVICE_SETUP_CHANGED, events.append)
    loop = QEventLoop()
    svc.setup_finished.connect(lambda _name: loop.quit())

    svc.start_setup_device(
        SetupDeviceRequest(name="test_dev", info=FakeDeviceInfo(address="none"))
    )
    loop.exec()

    assert events[0].active_setup is not None
    assert events[0].active_setup.device_name == "test_dev"
    assert events[-1].active_setup is None


# ---------------------------------------------------------------------------
# Registry CRUD (real FakeDevice through GlobalDeviceManager)
# ---------------------------------------------------------------------------


def _make_real_svc(driver: object | None = None) -> tuple[DeviceService, object]:
    fake_device = driver if driver is not None else FakeDevice()
    svc = DeviceService(
        EventBus(),
        State(MagicMock()),
        driver_factory=lambda _type, _address: fake_device,  # type: ignore[arg-type]
    )
    return svc, fake_device


def _register(svc: DeviceService, name: str = "flux") -> None:
    loop = QEventLoop()
    svc.device_connected.connect(lambda _request: loop.quit())
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name=name, address="")
    )
    loop.exec()


def _disconnect(svc: DeviceService, name: str = "flux") -> None:
    loop = QEventLoop()
    svc.device_disconnected.connect(lambda _request: loop.quit())
    svc.start_disconnect_device(DisconnectDeviceRequest(name=name))
    loop.exec()


def _set_value(svc: DeviceService, name: str, value: float) -> None:
    loop = QEventLoop()
    svc.value_set.connect(lambda _name: loop.quit())
    svc.start_set_device_value(SetDeviceValueRequest(name=name, value=value))
    loop.exec()


def test_devicemanager_register_and_list(qapp):
    dev = FakeDevice()
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")
    entries = svc.list_devices()
    entry = next((e for e in entries if e.name == "flux"), None)
    assert entry is not None
    assert entry.type_name == "FakeDevice"
    assert entry.is_connected is True


def test_devicemanager_drop_device(qapp):
    svc, _ = _make_real_svc()
    _register(svc, "flux")
    _disconnect(svc)
    # drop moves device to memory-only; it still appears but disconnected
    entries = svc.list_devices()
    flux_entry = next((e for e in entries if e.name == "flux"), None)
    assert flux_entry is not None
    assert not flux_entry.is_connected


def test_devicemanager_get_set_value(qapp):
    dev = FakeDevice()
    dev.set_value(3.14)
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")

    assert svc.get_device_value("flux") == pytest.approx(3.14)
    _set_value(svc, "flux", 2.71)
    assert svc.get_device_value("flux") == pytest.approx(2.71)


def test_devicemanager_get_all_info(qapp):
    dev = FakeDevice()
    dev.set_value(1.0)
    svc, _ = _make_real_svc(driver=dev)
    _register(svc, "flux")
    info = GlobalDeviceManager.get_all_info()
    assert "flux" in info
    flux_info = info["flux"]
    assert isinstance(flux_info, FakeDeviceInfo)
    assert flux_info.value == pytest.approx(1.0)
