"""Smoke tests for DeviceDialog."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from zcu_tools.gui.ui.device_dialog import DeviceDialog


def test_device_dialog_init(qapp):
    ctrl = MagicMock()
    ctrl.list_devices.return_value = {"fakedevice": "FakeDevice"}

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


def test_device_dialog_add_device(qapp):
    ctrl = MagicMock()
    ctrl.list_devices.return_value = {}

    with patch("zcu_tools.gui.ui.device_dialog._instantiate_device") as mock_inst:
        mock_inst.return_value = "mock_device_instance"

        dialog = DeviceDialog(ctrl)

        # Select FakeDevice and input address
        dialog._type_combo.setCurrentText("FakeDevice")
        dialog._addr_edit.setText("TCPIP::127.0.0.1::INSTR")

        # Click add button
        dialog._add_btn.click()

        mock_inst.assert_called_with("FakeDevice", "TCPIP::127.0.0.1::INSTR")
        ctrl.register_device.assert_called_with("fakedevice", "mock_device_instance")
        ctrl.setup_device.assert_called_with(
            "fakedevice", {"address": "TCPIP::127.0.0.1::INSTR"}
        )


def test_device_dialog_drop_device(qapp):
    ctrl = MagicMock()
    ctrl.list_devices.return_value = {"yoko": "YOKOGS200"}

    info_mock = MagicMock()
    info_mock.type = "YOKOGS200"
    ctrl.get_device_info.return_value = info_mock

    dialog = DeviceDialog(ctrl)

    assert dialog._list.count() == 1
    dialog._list.setCurrentRow(0)

    # Click drop
    dialog._drop_btn.click()
    ctrl.drop_device.assert_called_with("yoko")


def test_device_dialog_apply_changes(qapp):
    ctrl = MagicMock()
    ctrl.list_devices.return_value = {"sgs": "RohdeSchwarzSGS100A"}

    info_mock = MagicMock()
    info_mock.type = "RohdeSchwarzSGS100A"
    ctrl.get_device_info.return_value = info_mock

    dialog = DeviceDialog(ctrl)
    dialog._list.setCurrentRow(0)

    # Stack index should be 3 for SGS100A
    assert dialog._stack.currentIndex() == 3

    # Click apply
    dialog._apply_btn.click()

    # Should call set_device_value with extracted fields from panel
    ctrl.set_device_value.assert_called()
