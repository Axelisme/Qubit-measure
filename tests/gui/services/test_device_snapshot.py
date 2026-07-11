"""Tests for cached DeviceService render queries."""

from __future__ import annotations

import time
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.device.sgs100a import RohdeSchwarzSGS100AInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.app.main.services.operation_gate import OperationGate
from zcu_tools.gui.app.main.state import DeviceState, DeviceStatus, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.device import ConnectDeviceRequest, DeviceService
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui.services._device_fakes import FakeDeviceRegistry

# See tests/gui/services/test_device.py for why test-created BackgroundRunners must
# be quiesced before GC: a queued main-thread delivery to a GC'd runner segfaults.
_LIVE_BG: list[BackgroundRunner] = []


def _bg() -> BackgroundRunner:
    bg = BackgroundRunner()
    _LIVE_BG.append(bg)
    return bg


@pytest.fixture(autouse=True)
def _quiesce_services():
    yield
    for bg in _LIVE_BG:
        bg.quiesce()
    _LIVE_BG.clear()


@pytest.fixture(autouse=True)
def _clean_devices():
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)
    yield
    for name in list(GlobalDeviceManager.get_all_devices()):
        GlobalDeviceManager.drop_device(name)


def _drain_until(
    condition: Callable[[], bool], *, timeout: float = 3.0, label: str = "condition"
) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {label}")


def _make_svc(driver: object | None = None) -> DeviceService:
    fake = driver if driver is not None else FakeDevice()
    gate = OperationGate(EventBus())
    bg = _bg()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    runner = OperationRunner(gate, handles, progress, bg, bus)
    return DeviceService(
        bus,
        State(MagicMock()),
        gate,
        bg,
        runner,
        handles,
        driver_factory=lambda _type, _address: fake,  # type: ignore[arg-type]
        device_registry=FakeDeviceRegistry(),
    )


def _connect(
    svc: DeviceService, name: str = "flux", type_name: str = "FakeDevice"
) -> None:
    connected: list[object] = []
    errors: list[str] = []
    svc.device_connected.connect(connected.append)
    svc.operation_failed.connect(lambda _name, error: errors.append(error))
    svc.start_connect_device(
        ConnectDeviceRequest(type_name=type_name, name=name, address="")
    )
    _drain_until(lambda: bool(connected or errors), label=f"connect {name}")
    assert not errors


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


def test_get_cached_device_value_uses_state_without_reading_hardware(qapp):
    yoko_info = YOKOGS200Info(address="GPIB::1", mode="voltage", value=0.003)
    fake_yoko = MagicMock()
    fake_yoko.get_info.return_value = yoko_info
    svc = _make_svc(driver=fake_yoko)
    _connect(svc, type_name="YOKOGS200")
    fake_yoko.get_info.reset_mock()

    assert svc.get_cached_device_value("flux") == pytest.approx(0.003)
    fake_yoko.get_info.assert_not_called()


def test_get_cached_device_value_accepts_fake_device_info(qapp):
    svc = _make_svc()
    _connect(svc, "fake_flux", type_name="FakeDevice")

    assert svc.get_cached_device_value("fake_flux") == pytest.approx(0.0)


def test_get_cached_device_value_rejects_unsupported_device_type(qapp):
    fake_source = MagicMock()
    fake_source.get_info.return_value = RohdeSchwarzSGS100AInfo(
        address="TCPIP::1", freq_Hz=5e9, power_dBm=-20.0
    )
    svc = _make_svc(driver=fake_source)
    _connect(svc, "lo", type_name="RohdeSchwarzSGS100A")

    assert svc.get_cached_device_value("lo") is None


def test_get_cached_device_value_rejects_non_connected_or_missing_info(qapp):
    svc = _make_svc()
    state = svc._state
    state.put_device(
        DeviceState(
            name="memory",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.MEMORY_ONLY,
            remember=True,
            info=FakeDeviceInfo(address="none", value=1.0),
        )
    )
    state.put_device(
        DeviceState(
            name="setting",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.SETTING_UP,
            remember=True,
            info=FakeDeviceInfo(address="none", value=2.0),
        )
    )
    state.put_device(
        DeviceState(
            name="missing",
            type_name="FakeDevice",
            address="none",
            status=DeviceStatus.CONNECTED,
            remember=True,
            info=None,
        )
    )

    assert svc.get_cached_device_value("memory") is None
    assert svc.get_cached_device_value("setting") is None
    assert svc.get_cached_device_value("missing") is None


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
