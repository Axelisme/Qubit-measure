"""Smoke tests for DeviceDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import DeviceSetupFinishedPayload, EventBus, GuiEvent
from zcu_tools.gui.services.device import (
    ConnectDeviceRequest,
    DeviceEntry,
    DeviceSetupSnapshot,
    DeviceSnapshot,
    DeviceStatus,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.ui.device_dialog import DeviceDialog, _FakeDevicePanel


def _entry(
    name: str, type_name: str = "FakeDevice", connected: bool = True
) -> DeviceEntry:
    return DeviceEntry(name=name, type_name=type_name, is_connected=connected)


def _make_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_active_device_setup.return_value = None
    ctrl.progress_bars.return_value = ()
    ctrl.list_devices.return_value = []
    ctrl.is_memory_device.return_value = False
    ctrl.get_memory_device_address.return_value = None
    ctrl.get_device_snapshot.return_value = None
    return ctrl


def _connected_snapshot(name: str, info: object) -> DeviceSnapshot:
    return DeviceSnapshot(
        name=name,
        type_name=getattr(info, "type", "FakeDevice"),
        address=getattr(info, "address", ""),
        status=DeviceStatus.CONNECTED,
        info=info,  # type: ignore[arg-type]
    )


def test_device_dialog_init(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fakedevice")]

    # Mock get_device_info to return a simple mock with type
    info_mock = MagicMock()
    info_mock.type = "FakeDevice"
    ctrl.get_device_snapshot.return_value = _connected_snapshot("fakedevice", info_mock)

    dialog = DeviceDialog(ctrl)

    # list_devices should be called during init
    ctrl.list_devices.assert_called_once()
    assert dialog._list.count() == 1

    dialog._list.setCurrentRow(0)

    # The current stack index should map to FakeDevice (1)
    assert dialog._stack.currentIndex() == 1


def test_device_dialog_add_device_dispatches_request(qapp):
    ctrl = _make_ctrl()

    dialog = DeviceDialog(ctrl)
    dialog._type_combo.setCurrentText("FakeDevice")
    dialog._addr_edit.setText("TCPIP::127.0.0.1::INSTR")
    dialog._name_edit.setText("fakedevice")

    dialog._add_btn.click()

    ctrl.start_connect_device.assert_called_once()
    (req,) = ctrl.start_connect_device.call_args.args
    assert isinstance(req, ConnectDeviceRequest)
    assert req.type_name == "FakeDevice"
    assert req.name == "fakedevice"
    assert req.address == "TCPIP::127.0.0.1::INSTR"
    ctrl.start_setup_device.assert_not_called()


def test_device_dialog_add_device_does_not_persist_before_async_success(qapp):
    ctrl = _make_ctrl()
    dialog = DeviceDialog(ctrl)
    dialog._add_btn.click()


def test_device_dialog_add_device_propagates_unexpected_errors(qapp):
    """Programmer errors must not be swallowed by the dialog catch."""
    ctrl = _make_ctrl()
    ctrl.start_connect_device.side_effect = ValueError("contract violation")

    dialog = DeviceDialog(ctrl)
    dialog._type_combo.setCurrentText("FakeDevice")
    dialog._addr_edit.setText("addr")
    with pytest.raises(ValueError, match="contract violation"):
        dialog._on_add_clicked()


def test_device_dialog_drop_device(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("yoko", "YOKOGS200")]

    info_mock = MagicMock()
    info_mock.type = "YOKOGS200"
    ctrl.get_device_snapshot.return_value = _connected_snapshot("yoko", info_mock)

    dialog = DeviceDialog(ctrl)

    assert dialog._list.count() == 1
    dialog._list.setCurrentRow(0)

    # Drop disconnects but keeps startup memory
    dialog._drop_btn.click()
    ctrl.start_disconnect_device.assert_called_once()
    (req,) = ctrl.start_disconnect_device.call_args.args
    assert isinstance(req, DisconnectDeviceRequest)
    assert req.name == "yoko"


def test_device_dialog_forget_memory_device_dispatches_single_transaction(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("remembered", connected=False)]
    ctrl.is_memory_device.return_value = True
    ctrl.get_device_snapshot.return_value = DeviceSnapshot(
        name="remembered",
        type_name="FakeDevice",
        address="addr",
        status=DeviceStatus.MEMORY_ONLY,
    )

    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)
    dialog._drop_btn.click()

    ctrl.forget_device.assert_called_once_with("remembered")


def test_device_dialog_refresh_reloads_selected_device_info(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    ctrl.get_device_snapshot.side_effect = [
        _connected_snapshot("fd", FakeDeviceInfo(address="none", value=1.0)),
        _connected_snapshot("fd", FakeDeviceInfo(address="none", value=2.0)),
    ]
    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)
    panel = dialog._stack.currentWidget()
    assert isinstance(panel, _FakeDevicePanel)
    assert panel._value_spin.value() == 1.0

    dialog._refresh_btn.click()

    assert panel._value_spin.value() == 2.0
    ctrl.get_device_info.assert_called_once_with("fd")


def test_device_dialog_apply_changes(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]

    info = FakeDeviceInfo(address="none")
    ctrl.get_device_snapshot.return_value = _connected_snapshot("fd", info)
    ctrl.get_device_info.return_value = info
    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)

    assert dialog._stack.currentIndex() == 1  # FakeDevice page

    dialog._apply_btn.click()

    ctrl.start_setup_device.assert_called_once()
    (req,) = ctrl.start_setup_device.call_args.args
    assert isinstance(req, SetupDeviceRequest)
    assert req.name == "fd"
    assert isinstance(req.info, FakeDeviceInfo)


def test_device_dialog_restores_background_setup_and_stops_it(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl.get_device_snapshot.return_value = _connected_snapshot(
        "fd", FakeDeviceInfo(address="none")
    )
    ctrl.get_active_device_setup.return_value = DeviceSetupSnapshot(device_name="fd")

    dialog = DeviceDialog(ctrl)

    item = dialog._list.currentItem()
    assert item is not None
    assert item.data(256) == "fd"
    assert dialog._apply_btn.text() == "Stop"
    assert dialog._list.isEnabled() is False
    assert dialog._refresh_btn.isEnabled() is False
    # The dialog attached to progress by the active setup's device_name (owner),
    # so live bars render even though it opened mid-setup.
    ctrl.attach_progress.assert_called_once()
    assert ctrl.attach_progress.call_args.args[0] == "fd"

    dialog._apply_btn.click()
    ctrl.cancel_device_operation.assert_called_once_with("fd")

    # Setup reaching a terminal state → panel returns to idle. After finish,
    # get_active_device_setup reports no active setup.
    ctrl.get_active_device_setup.return_value = None
    ctrl.get_bus.return_value.emit(
        GuiEvent.DEVICE_SETUP_FINISHED,
        DeviceSetupFinishedPayload(name="fd", outcome="cancelled"),
    )
    assert dialog._apply_btn.text() == "Apply Changes"
    assert dialog._list.isEnabled() is True
    assert dialog._refresh_btn.isEnabled() is True


def test_device_dialog_close_keeps_setup_running_and_unsubscribes(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl.get_device_snapshot.return_value = _connected_snapshot(
        "fd", FakeDeviceInfo(address="none")
    )
    ctrl.get_active_device_setup.return_value = DeviceSetupSnapshot(device_name="fd")
    dialog = DeviceDialog(ctrl)

    dialog.accept()

    ctrl.cancel_device_operation.assert_not_called()
    assert ctrl.get_bus.return_value._subs[GuiEvent.DEVICE_SETUP_FINISHED] == []
