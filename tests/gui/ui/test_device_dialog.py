"""Smoke tests for DeviceDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import DeviceSetupChangedPayload, EventBus, GuiEvent
from zcu_tools.gui.services.device import (
    DeviceEntry,
    DeviceRegistrationError,
    DeviceSetupSnapshot,
    RegisterDeviceRequest,
)
from zcu_tools.gui.services.device_progress import ProgressEntrySnapshot
from zcu_tools.gui.ui.device_dialog import DeviceDialog, _FakeDevicePanel


def _entry(
    name: str, type_name: str = "FakeDevice", connected: bool = True
) -> DeviceEntry:
    return DeviceEntry(name=name, type_name=type_name, is_connected=connected)


def _make_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_active_device_setup.return_value = None
    ctrl.list_devices.return_value = []
    ctrl.is_memory_device.return_value = False
    ctrl.get_memory_device_address.return_value = None
    return ctrl


def test_device_dialog_init(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fakedevice")]

    # Mock get_device_info to return a simple mock with type
    info_mock = MagicMock()
    info_mock.type = "FakeDevice"
    ctrl.get_device_info.return_value = info_mock

    dialog = DeviceDialog(ctrl)

    # list_devices should be called during init
    ctrl.list_devices.assert_called_once()
    assert dialog._list.count() == 1

    dialog._list.setCurrentRow(0)

    # Since an item is added and selected, get_device_info should be called
    ctrl.get_device_info.assert_called_with("fakedevice")

    # The current stack index should map to FakeDevice (1)
    assert dialog._stack.currentIndex() == 1


def test_device_dialog_add_device_dispatches_request(qapp):
    ctrl = _make_ctrl()

    dialog = DeviceDialog(ctrl)
    dialog._type_combo.setCurrentText("FakeDevice")
    dialog._addr_edit.setText("TCPIP::127.0.0.1::INSTR")
    dialog._name_edit.setText("fakedevice")

    dialog._add_btn.click()

    ctrl.register_device.assert_called_once()
    (req,) = ctrl.register_device.call_args.args
    assert isinstance(req, RegisterDeviceRequest)
    assert req.type_name == "FakeDevice"
    assert req.name == "fakedevice"
    assert req.address == "TCPIP::127.0.0.1::INSTR"
    ctrl.setup_device.assert_not_called()


def test_device_dialog_add_device_shows_registration_error(qapp):
    ctrl = _make_ctrl()
    ctrl.register_device.side_effect = DeviceRegistrationError("boom")

    dialog = DeviceDialog(ctrl)
    dialog._type_combo.setCurrentText("FakeDevice")
    dialog._addr_edit.setText("addr")
    dialog._add_btn.click()

    assert "boom" in dialog._add_status.text()


def test_device_dialog_add_device_propagates_unexpected_errors(qapp):
    """Programmer errors must not be swallowed by the dialog catch."""
    ctrl = _make_ctrl()
    ctrl.register_device.side_effect = ValueError("contract violation")

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
    ctrl.get_device_info.return_value = info_mock

    dialog = DeviceDialog(ctrl)

    assert dialog._list.count() == 1
    dialog._list.setCurrentRow(0)

    # Click forget (was drop)
    dialog._drop_btn.click()
    ctrl.drop_device.assert_called_with("yoko")


def test_device_dialog_refresh_reloads_selected_device_info(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    ctrl.get_device_info.side_effect = [
        FakeDeviceInfo(address="none", value=1.0),
        FakeDeviceInfo(address="none", value=2.0),
    ]
    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)
    panel = dialog._stack.currentWidget()
    assert isinstance(panel, _FakeDevicePanel)
    assert panel._value_spin.value() == 1.0

    dialog._refresh_btn.click()

    assert panel._value_spin.value() == 2.0
    assert ctrl.get_device_info.call_count == 2


def test_device_dialog_apply_changes(qapp):
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]

    info = FakeDeviceInfo(address="none")
    ctrl.get_device_info.return_value = info
    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)

    assert dialog._stack.currentIndex() == 1  # FakeDevice page

    dialog._apply_btn.click()

    # Should call setup_device with a FakeDeviceInfo built from with_updates
    ctrl.setup_device.assert_called_once()
    call_args = ctrl.setup_device.call_args
    assert call_args.args[0] == "fd"
    assert isinstance(call_args.args[1], FakeDeviceInfo)
    assert len(call_args.args) == 2


def test_device_dialog_restores_background_setup_and_stops_it(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl.get_device_info.return_value = FakeDeviceInfo(address="none")
    snapshot = DeviceSetupSnapshot(
        device_name="fd",
        progress=(
            ProgressEntrySnapshot(
                token=1, format="Ramp value [0:01]", maximum=10000, value=5000
            ),
        ),
    )
    ctrl.get_active_device_setup.return_value = snapshot

    dialog = DeviceDialog(ctrl)

    item = dialog._list.currentItem()
    assert item is not None
    assert item.data(256) == "fd"
    assert dialog._apply_btn.text() == "Stop"
    assert dialog._list.isEnabled() is False
    assert dialog._refresh_btn.isEnabled() is False
    assert dialog._progress._active[0].value() == 5000

    dialog._apply_btn.click()
    ctrl.cancel_device_setup.assert_called_once_with()

    ctrl.get_bus.return_value.emit(
        GuiEvent.DEVICE_SETUP_CHANGED, DeviceSetupChangedPayload(active_setup=None)
    )
    assert dialog._apply_btn.text() == "Apply Changes"
    assert dialog._list.isEnabled() is True
    assert dialog._refresh_btn.isEnabled() is True


def test_device_dialog_close_keeps_setup_running_and_unsubscribes(qapp):
    ctrl = _make_ctrl()
    ctrl.list_devices.return_value = [_entry("fd")]
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl.get_device_info.return_value = FakeDeviceInfo(address="none")
    ctrl.get_active_device_setup.return_value = DeviceSetupSnapshot(
        device_name="fd", progress=()
    )
    dialog = DeviceDialog(ctrl)

    dialog.accept()

    ctrl.cancel_device_setup.assert_not_called()
    assert ctrl.get_bus.return_value._subs[GuiEvent.DEVICE_SETUP_CHANGED] == []
