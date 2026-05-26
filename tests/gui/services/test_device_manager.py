"""Tests for device setup worker lifecycle and progress events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.event_bus import DeviceSetupChangedPayload, EventBus, GuiEvent
from zcu_tools.gui.services.device import (
    ConnectDeviceRequest,
    DeviceService,
    SetupDeviceRequest,
    _DeviceSetupWorker,
)


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
        driver_factory=lambda _type, _address: device,
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
