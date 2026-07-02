"""DeviceControlFacet delegation and event-subscription contract."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.device_control import DeviceControlFacet
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)


def _facet() -> tuple[DeviceControlFacet, MagicMock, MagicMock, BaseEventBus]:
    bus = BaseEventBus()
    device = MagicMock()
    progress = MagicMock()
    return (
        DeviceControlFacet(
            bus=bus,
            device=cast(Any, device),
            progress=cast(Any, progress),
        ),
        device,
        progress,
        bus,
    )


def test_device_control_facet_delegates_device_and_progress_calls() -> None:
    facet, device, progress, _bus = _facet()

    connect_req = ConnectDeviceRequest("FakeDevice", "fd", "addr")
    disconnect_req = DisconnectDeviceRequest("fd")
    setup_req = SetupDeviceRequest("fd", FakeDeviceInfo(address="addr"))
    listener = MagicMock()

    device.start_connect_device.return_value = 11
    device.start_disconnect_device.return_value = 12
    device.start_reconnect_device.return_value = 13
    device.start_setup_device.return_value = 14
    device.list_devices.return_value = ["entry"]
    device.get_device_snapshot.return_value = "snapshot"
    device.get_device_info.return_value = "info"
    device.is_memory_device.return_value = True
    device.get_device_unit.return_value = "A"
    device.get_active_device_operations.return_value = ("op",)
    progress.attach_by_owner.return_value = "dispose"
    progress.bars_for_owner.return_value = ((1, "bar"),)

    assert facet.start_connect_device(connect_req) == 11
    assert facet.start_disconnect_device(disconnect_req) == 12
    assert facet.start_reconnect_device("fd") == 13
    assert facet.start_setup_device(setup_req) == 14
    facet.forget_device("fd")
    facet.cancel_device_operation("fd")
    assert facet.list_devices() == ["entry"]
    assert facet.get_device_snapshot("fd") == "snapshot"
    assert facet.get_device_info("fd") == "info"
    facet.poll_device_info("fd")
    assert facet.is_memory_device("fd") is True
    assert facet.get_device_unit("fd") == "A"
    assert facet.get_active_device_operations() == ("op",)
    assert facet.attach_progress("fd", listener) == "dispose"
    assert facet.progress_bars("fd") == ((1, "bar"),)

    device.start_connect_device.assert_called_once_with(connect_req)
    device.start_disconnect_device.assert_called_once_with(disconnect_req)
    device.start_reconnect_device.assert_called_once_with("fd")
    device.start_setup_device.assert_called_once_with(setup_req)
    device.forget_device.assert_called_once_with("fd")
    device.cancel_device_operation.assert_called_once_with("fd")
    device.list_devices.assert_called_once_with()
    device.get_device_snapshot.assert_called_once_with("fd")
    device.get_device_info.assert_called_once_with("fd")
    device.poll_device_info.assert_called_once_with("fd")
    device.is_memory_device.assert_called_once_with("fd")
    device.get_device_unit.assert_called_once_with("fd")
    device.get_active_device_operations.assert_called_once_with()
    progress.attach_by_owner.assert_called_once_with("fd", listener)
    progress.bars_for_owner.assert_called_once_with("fd")


def test_device_control_facet_event_disposer_unsubscribes() -> None:
    facet, _device, _progress, bus = _facet()
    changed: list[str | None] = []
    started: list[str] = []
    finished: list[str] = []

    unsubscribe_changed = facet.on_device_changed(lambda p: changed.append(p.name))
    unsubscribe_started = facet.on_device_setup_started(
        lambda p: started.append(p.name)
    )
    unsubscribe_finished = facet.on_device_setup_finished(
        lambda p: finished.append(p.name)
    )

    bus.emit(DeviceChangedPayload(name="fd"))
    bus.emit(DeviceSetupStartedPayload(name="fd"))
    bus.emit(DeviceSetupFinishedPayload(name="fd", outcome="finished"))
    assert changed == ["fd"]
    assert started == ["fd"]
    assert finished == ["fd"]

    unsubscribe_changed()
    unsubscribe_started()
    unsubscribe_finished()
    unsubscribe_changed()
    unsubscribe_started()
    unsubscribe_finished()
    bus.emit(DeviceChangedPayload(name="fd2"))
    bus.emit(DeviceSetupStartedPayload(name="fd2"))
    bus.emit(DeviceSetupFinishedPayload(name="fd2", outcome="finished"))
    assert changed == ["fd"]
    assert started == ["fd"]
    assert finished == ["fd"]
