"""Tests for cached DeviceService render queries."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.device.sgs100a import RohdeSchwarzSGS100AInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.state import DeviceState, DeviceStatus, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.device_errors import DeviceRegistrationError
from zcu_tools.gui.session.device_models import ConnectDeviceRequest
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import (
    OperationInfrastructure,
    OperationRunner,
)
from zcu_tools.gui.session.ports import DeviceMemoryInfo, DeviceProtocol
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui._completion_helpers import (
    on_device_connected,
    on_device_operation_failed,
)
from tests.gui._qt_waits import qt_until
from tests.gui.session.services._device_fakes import FakeDeviceRegistry
from tests.gui.session.services._device_waits import DeviceWorkers

_workers = DeviceWorkers()
_bg = _workers.background


@pytest.fixture(autouse=True, name="_quiesce_services")
def fixture_quiesce_services():
    """Close tracked service workers after each test."""
    try:
        yield
    finally:
        _workers.close()


def _make_svc(
    driver: DeviceProtocol | None = None, state: State | None = None
) -> DeviceService:
    fake = driver if driver is not None else FakeDevice()
    gate = OperationGate(EventBus())
    bg = _bg()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    bus = EventBus()
    _workers.watch(bus)
    runner = OperationRunner(gate, handles, progress, bg, bus)
    return DeviceService(
        bus,
        state or State(MagicMock()),
        OperationInfrastructure(
            gate=gate,
            background=bg,
            runner=runner,
            handles=handles,
        ),
        driver_factory=lambda type_name, address: fake,
        device_registry=FakeDeviceRegistry(),
    )


def _connect(
    svc: DeviceService, name: str = "flux", type_name: str = "FakeDevice"
) -> None:
    connected: list[object] = []
    errors: list[str] = []
    on_device_connected(svc, connected.append)
    on_device_operation_failed(svc, lambda _name, error: errors.append(error))
    svc.start_connect_device(
        ConnectDeviceRequest(type_name=type_name, name=name, address="")
    )
    qt_until(lambda: bool(connected or errors), label=f"connect {name}")
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
    state = State(MagicMock())
    svc = _make_svc(state=state)
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


def test_cached_fake_unit_is_native(qapp) -> None:
    svc = _make_svc(driver=FakeDevice(fast_mode=True))
    _connect(svc)
    before = svc.list_device_snapshots()

    assert svc.get_device_unit("flux") == "native"
    assert svc.get_device_unit_strict("flux") == "native"
    info = before[0].info
    assert info is not None
    assert info.model_dump()["value_unit"] == "native"
    assert svc.list_device_snapshots() == before


@pytest.mark.parametrize("type_name", ["YOKOGS200", "FakeDevice"])
def test_remembered_device_without_info_has_no_inferred_unit(
    qapp, type_name: str
) -> None:
    svc = _make_svc()
    svc.register_remembered_devices(
        [DeviceMemoryInfo(name="flux", type_name=type_name, address="unused")]
    )
    before = svc.list_device_snapshots()
    assert before[0].info is None
    assert before[0].status == DeviceStatus.MEMORY_ONLY

    assert svc.get_device_unit("flux") == "none"
    with pytest.raises(DeviceRegistrationError, match="flux"):
        svc.get_device_unit_strict("flux")
    assert svc.list_device_snapshots() == before
