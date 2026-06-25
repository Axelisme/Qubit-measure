"""Tests for DeviceRefSpec, DeviceRefLiveField, and device change events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QEventLoop
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.app.main.adapter import DeviceRefSpec, DirectValue
from zcu_tools.gui.app.main.live_model import DeviceRefLiveField, LiveModelEnv
from zcu_tools.gui.app.main.services.operation_gate import OperationGate
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.background import BackgroundRunner
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.events import DeviceChangedPayload, SessionEvent
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DeviceService,
    DisconnectDeviceRequest,
)
from zcu_tools.gui.session.services.progress import ProgressService

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


def _make_env(device_names: list[str] | None = None) -> LiveModelEnv:
    ctrl = MagicMock()
    ctrl.list_device_names.return_value = list(device_names or [])
    return LiveModelEnv(ctrl=ctrl)


def _make_field(
    initial_name: str = "", device_names: list[str] | None = None
) -> DeviceRefLiveField:
    spec = DeviceRefSpec(label="Flux Device")
    env = _make_env(device_names)
    initial = DirectValue(initial_name) if initial_name else None
    return DeviceRefLiveField(spec, env, initial)


def _make_service(device: MagicMock) -> tuple[DeviceService, EventBus]:
    bus = EventBus()
    gate = OperationGate()
    bg = _bg()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    runner = OperationRunner(gate, handles, progress, bg)
    return (
        DeviceService(
            bus,
            State(MagicMock()),
            gate,
            bg,
            runner,
            handles,
            driver_factory=lambda _type, _address: device,  # type: ignore[arg-type]
        ),
        bus,
    )


def test_device_ref_valid_when_device_exists():
    field = _make_field("flux_yoko", device_names=["flux_yoko"])
    assert field.is_valid()
    assert field.get_chosen_name() == "flux_yoko"


def test_device_ref_invalid_when_device_missing():
    assert not _make_field("flux_yoko", device_names=[]).is_valid()


def test_device_ref_invalid_when_empty():
    assert not _make_field("", device_names=["flux_yoko"]).is_valid()


def test_set_chosen_name_emits_on_change():
    events: list = []
    field = _make_field("dev_a", device_names=["dev_a", "dev_b"])
    field.on_change.connect(events.append)
    field.set_chosen_name("dev_b")
    assert len(events) == 1
    assert isinstance(events[0], DirectValue)
    assert events[0].value == "dev_b"


def test_set_value_direct_value():
    field = _make_field("", device_names=["dev_a"])
    field.set_value(DirectValue("dev_a"))
    assert field.get_chosen_name() == "dev_a"


def test_set_value_invalid_type_raises():
    field = _make_field("", device_names=[])
    with pytest.raises(TypeError):
        field.set_value(42)


def test_refresh_external_device_changed_updates_validity():
    validity_events: list[bool] = []
    field = _make_field("flux_yoko", device_names=[])
    field.on_validity_changed.connect(validity_events.append)
    field.env.ctrl.list_device_names.return_value = ["flux_yoko"]  # type: ignore[attr-defined]

    field.refresh_external(SessionEvent.DEVICE_CHANGED)

    assert field.is_valid()
    assert True in validity_events


def test_device_service_emits_pending_and_connected_events(qapp):
    device = MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="")
    svc, bus = _make_service(device)
    received: list[DeviceChangedPayload] = []
    bus.subscribe(DeviceChangedPayload, received.append)
    loop = QEventLoop()
    svc.device_connected.connect(lambda _request: loop.quit())

    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name="dev1", address="")
    )
    loop.exec()

    assert [payload.name for payload in received] == ["dev1", "dev1"]


def test_device_service_emits_pending_and_disconnected_events(qapp):
    device = MagicMock()
    device.get_info.return_value = FakeDeviceInfo(address="")
    svc, bus = _make_service(device)
    connect_loop = QEventLoop()
    svc.device_connected.connect(lambda _request: connect_loop.quit())
    svc.start_connect_device(
        ConnectDeviceRequest(type_name="FakeDevice", name="dev1", address="")
    )
    connect_loop.exec()
    received: list[DeviceChangedPayload] = []
    bus.subscribe(DeviceChangedPayload, received.append)
    loop = QEventLoop()
    svc.device_disconnected.connect(lambda _request: loop.quit())

    svc.start_disconnect_device(DisconnectDeviceRequest(name="dev1"))
    loop.exec()

    device.close.assert_called_once_with()
    assert [payload.name for payload in received] == ["dev1", "dev1"]
