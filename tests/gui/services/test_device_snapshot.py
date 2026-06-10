"""Tests for cached DeviceService render queries."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDevice
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.app.main.event_bus import EventBus
from zcu_tools.gui.app.main.services.background import BackgroundService
from zcu_tools.gui.app.main.services.operation_gate import OperationGate
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.services.device import ConnectDeviceRequest, DeviceService
from zcu_tools.gui.session.services.progress import ProgressService


@pytest.fixture(autouse=True)
def _clean_devices():
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)
    yield
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)


def _make_svc(driver: object | None = None) -> DeviceService:
    fake = driver if driver is not None else FakeDevice()
    return DeviceService(
        EventBus(),
        State(MagicMock()),
        OperationGate(),
        BackgroundService(),
        ProgressService(QtProgressTransport()),
        driver_factory=lambda _type, _address: fake,  # type: ignore[arg-type]
    )


def _connect(
    svc: DeviceService, name: str = "flux", type_name: str = "FakeDevice"
) -> None:
    loop = QEventLoop()
    svc.device_connected.connect(lambda _request: loop.quit())
    svc.start_connect_device(
        ConnectDeviceRequest(type_name=type_name, name=name, address="")
    )
    loop.exec()


def test_list_device_names_uses_connected_snapshot(qapp):
    svc = _make_svc()
    _connect(svc, "z")
    svc.register_remembered_devices([])
    assert svc.list_device_names() == ["z"]


def test_get_device_unit_for_cached_yoko_info(qapp):
    yoko_info = YOKOGS200Info(address="GPIB::1", mode="voltage", value=0.003)
    fake_yoko = MagicMock()
    fake_yoko.get_info.return_value = yoko_info
    svc = _make_svc(driver=fake_yoko)

    _connect(svc, type_name="YOKOGS200")

    assert svc.get_device_unit("flux") == "V"
    assert svc.get_device_value_for_new_context("flux") == 0.003


def test_get_device_unit_yoko_current_mode_returns_a(qapp):
    """YOKOGS200 in current mode → "A" (default / non-voltage path)."""
    yoko_info = YOKOGS200Info(address="GPIB::1", mode="current", value=0.001)
    fake_yoko = MagicMock()
    fake_yoko.get_info.return_value = yoko_info
    svc = _make_svc(driver=fake_yoko)
    _connect(svc, type_name="YOKOGS200")
    assert svc.get_device_unit("flux") == "A"


def test_get_device_unit_yoko_voltage_mode_returns_v_strict(qapp):
    """YOKOGS200 voltage mode → "V" through the strict path too."""
    yoko_info = YOKOGS200Info(address="GPIB::1", mode="voltage", value=0.003)
    fake_yoko = MagicMock()
    fake_yoko.get_info.return_value = yoko_info
    svc = _make_svc(driver=fake_yoko)
    _connect(svc, type_name="YOKOGS200")
    assert svc.get_device_unit_strict("flux") == "V"


def test_get_device_unit_yoko_current_mode_returns_a_strict(qapp):
    """YOKOGS200 current mode → "A" through the strict path."""
    yoko_info = YOKOGS200Info(address="GPIB::1", mode="current", value=0.001)
    fake_yoko = MagicMock()
    fake_yoko.get_info.return_value = yoko_info
    svc = _make_svc(driver=fake_yoko)
    _connect(svc, type_name="YOKOGS200")
    assert svc.get_device_unit_strict("flux") == "A"


def test_list_snapshots_does_not_read_hardware(qapp):
    fake = MagicMock()
    fake.get_info.return_value = YOKOGS200Info(
        address="GPIB::1", mode="current", value=0.003
    )
    svc = _make_svc(driver=fake)
    _connect(svc)
    fake.get_info.reset_mock()

    snapshots = svc.list_device_snapshots()

    assert snapshots[0].name == "flux"
    fake.get_info.assert_not_called()
