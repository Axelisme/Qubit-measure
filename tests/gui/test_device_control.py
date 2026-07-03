"""DeviceControlFacet public contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.device_control import DeviceControlFacet
from zcu_tools.gui.session.events import (
    DeviceChangedPayload,
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.pbar_host import ProgressBarModel
from zcu_tools.gui.session.ports import OperationKind
from zcu_tools.gui.session.services.device import (
    ActiveDeviceOperation,
    ConnectDeviceRequest,
    DeviceEntry,
    DeviceSnapshot,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.state import DeviceStatus

from tests.gui._control_fakes import CallLog, RecordedCall, call


class RecordingDevice:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.info = FakeDeviceInfo(address="addr")
        self.snapshot = DeviceSnapshot(
            name="fd",
            type_name="FakeDevice",
            address="addr",
            status=DeviceStatus.CONNECTED,
            info=self.info,
        )
        self.active_operation = ActiveDeviceOperation(
            device_name="fd",
            kind=OperationKind.DEVICE_CONNECT,
            snapshot=self.snapshot,
            token=21,
        )

    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        self._log.add("device", "start_connect_device", req)
        return 11

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        self._log.add("device", "start_disconnect_device", req)
        return 12

    def start_reconnect_device(self, name: str) -> int:
        self._log.add("device", "start_reconnect_device", name)
        return 13

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        self._log.add("device", "start_setup_device", req)
        return 14

    def forget_device(self, name: str) -> None:
        self._log.add("device", "forget_device", name)

    def cancel_device_operation(self, name: str) -> None:
        self._log.add("device", "cancel_device_operation", name)

    def list_devices(self) -> list[DeviceEntry]:
        self._log.add("device", "list_devices")
        return [DeviceEntry(name="fd", type_name="FakeDevice", status="connected")]

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        self._log.add("device", "get_device_snapshot", name)
        return self.snapshot

    def get_device_info(self, name: str) -> FakeDeviceInfo | None:
        self._log.add("device", "get_device_info", name)
        return self.info

    def get_cached_device_value(self, name: str) -> float | None:
        self._log.add("device", "get_cached_device_value", name)
        return 0.125

    def poll_device_info(self, name: str) -> None:
        self._log.add("device", "poll_device_info", name)

    def is_memory_device(self, name: str) -> bool:
        self._log.add("device", "is_memory_device", name)
        return True

    def get_device_unit(self, name: str) -> str:
        self._log.add("device", "get_device_unit", name)
        return "A"

    def get_active_device_operations(self) -> tuple[ActiveDeviceOperation, ...]:
        self._log.add("device", "get_active_device_operations")
        return (self.active_operation,)


class RecordingProgress:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.dispose = _noop
        self.bar = ProgressBarModel("setup", 10, 0.0)

    def attach_by_owner(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        self._log.add("progress", "attach_by_owner", owner_id, listener)
        return self.dispose

    def bars_for_owner(self, owner_id: str) -> tuple[tuple[int, ProgressBarModel], ...]:
        self._log.add("progress", "bars_for_owner", owner_id)
        return ((1, self.bar),)


def _noop() -> None:
    pass


def _facet() -> tuple[
    DeviceControlFacet, CallLog, RecordingDevice, RecordingProgress, BaseEventBus
]:
    log = CallLog()
    bus = BaseEventBus()
    device = RecordingDevice(log)
    progress = RecordingProgress(log)
    return (
        DeviceControlFacet(
            bus=bus,
            device=cast(Any, device),
            progress=cast(Any, progress),
        ),
        log,
        device,
        progress,
        bus,
    )


def test_device_control_facet_forwards_deliberate_device_dialog_contract() -> None:
    facet, log, device, progress, _bus = _facet()
    connect_req = ConnectDeviceRequest("FakeDevice", "fd", "addr")
    disconnect_req = DisconnectDeviceRequest("fd")
    setup_req = SetupDeviceRequest("fd", FakeDeviceInfo(address="addr"))

    cases: tuple[tuple[str, Callable[[], object], object, RecordedCall], ...] = (
        (
            "start_connect_device",
            lambda: facet.start_connect_device(connect_req),
            11,
            call("device", "start_connect_device", connect_req),
        ),
        (
            "start_disconnect_device",
            lambda: facet.start_disconnect_device(disconnect_req),
            12,
            call("device", "start_disconnect_device", disconnect_req),
        ),
        (
            "start_reconnect_device",
            lambda: facet.start_reconnect_device("fd"),
            13,
            call("device", "start_reconnect_device", "fd"),
        ),
        (
            "start_setup_device",
            lambda: facet.start_setup_device(setup_req),
            14,
            call("device", "start_setup_device", setup_req),
        ),
        (
            "forget_device",
            lambda: facet.forget_device("fd"),
            None,
            call("device", "forget_device", "fd"),
        ),
        (
            "cancel_device_operation",
            lambda: facet.cancel_device_operation("fd"),
            None,
            call("device", "cancel_device_operation", "fd"),
        ),
        (
            "list_devices",
            facet.list_devices,
            [DeviceEntry(name="fd", type_name="FakeDevice", status="connected")],
            call("device", "list_devices"),
        ),
        (
            "get_device_snapshot",
            lambda: facet.get_device_snapshot("fd"),
            device.snapshot,
            call("device", "get_device_snapshot", "fd"),
        ),
        (
            "get_device_info",
            lambda: facet.get_device_info("fd"),
            device.info,
            call("device", "get_device_info", "fd"),
        ),
        (
            "get_cached_device_value",
            lambda: facet.get_cached_device_value("fd"),
            0.125,
            call("device", "get_cached_device_value", "fd"),
        ),
        (
            "poll_device_info",
            lambda: facet.poll_device_info("fd"),
            None,
            call("device", "poll_device_info", "fd"),
        ),
        (
            "is_memory_device",
            lambda: facet.is_memory_device("fd"),
            True,
            call("device", "is_memory_device", "fd"),
        ),
        (
            "get_device_unit",
            lambda: facet.get_device_unit("fd"),
            "A",
            call("device", "get_device_unit", "fd"),
        ),
        (
            "get_active_device_operations",
            facet.get_active_device_operations,
            (device.active_operation,),
            call("device", "get_active_device_operations"),
        ),
        (
            "attach_progress",
            lambda: facet.attach_progress("fd", _noop),
            progress.dispose,
            call("progress", "attach_by_owner", "fd", _noop),
        ),
        (
            "progress_bars",
            lambda: facet.progress_bars("fd"),
            ((1, progress.bar),),
            call("progress", "bars_for_owner", "fd"),
        ),
    )

    for name, action, expected_result, _expected_call in cases:
        assert action() == expected_result, name

    assert log.calls == [expected_call for *_, expected_call in cases]


def test_device_control_facet_event_disposer_unsubscribes() -> None:
    facet, _log, _device, _progress, bus = _facet()
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
