"""Keyboard submission, close results and destructive device confirmations."""

from pathlib import Path

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QMessageBox
from zcu_tools.gui.session.device_models import DeviceSnapshot, DisconnectDeviceRequest
from zcu_tools.gui.session.state import DeviceStatus

from tests.gui._user_input import ComponentDriver
from tests.gui.session.ui._device_dialog_support import (
    connected_snapshot,
    device_entry,
    make_device_control,
    make_device_dialog,
)


def test_close_rejects_without_accepting_changes(qapp, tmp_path: Path) -> None:
    ctrl = make_device_control()
    dialog = make_device_dialog(ctrl)
    try:
        dialog.show()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.screenshot(tmp_path / "device-before-close.png")
        driver.click("Close")
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert not dialog.isVisible()
    finally:
        dialog.reject()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("memory", [False, True])
def test_removing_device_requires_explicit_confirmation(
    qapp, tmp_path: Path, memory: bool
) -> None:
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("bias", connected=not memory)]
    ctrl.is_memory_device.return_value = memory
    ctrl.get_device_snapshot.return_value = DeviceSnapshot(
        name="bias",
        type_name="FakeDevice",
        address="none",
        status=DeviceStatus.MEMORY_ONLY if memory else DeviceStatus.CONNECTED,
        info=FakeDeviceInfo(address="none"),
    )
    dialog = make_device_dialog(ctrl)
    action = "Forget" if memory else "Drop"
    try:
        dialog.show()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.select_list_row(driver.list_rows()[0].text)
        driver.click(action)
        ctrl.forget_device.assert_not_called()
        ctrl.start_disconnect_device.assert_not_called()
        prompt = next(
            box for box in dialog.findChildren(QMessageBox) if box.isVisible()
        )
        question = ComponentDriver(prompt)
        question.screenshot(tmp_path / f"{action}-confirmation.png")
        question.press_key(Qt.Key.Key_Return)
        ctrl.forget_device.assert_not_called()
        ctrl.start_disconnect_device.assert_not_called()
        driver.click(action)
        prompt = next(
            box for box in dialog.findChildren(QMessageBox) if box.isVisible()
        )
        ComponentDriver(prompt).click(action)
        if memory:
            ctrl.forget_device.assert_called_once_with("bias")
        else:
            ctrl.start_disconnect_device.assert_called_once_with(
                DisconnectDeviceRequest(name="bias")
            )
    finally:
        for box in dialog.findChildren(QMessageBox):
            box.reject()
        dialog.reject()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("change", ["busy", "missing", "read-only", "closed"])
def test_pending_confirmation_rechecks_target_and_dialog_lifetime(
    qapp, change: str
) -> None:
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("bias")]
    ctrl.get_device_snapshot.return_value = connected_snapshot(
        "bias", FakeDeviceInfo(address="none")
    )
    dialog = make_device_dialog(ctrl)
    try:
        dialog.show()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.select_list_row(driver.list_rows()[0].text)
        driver.click("Drop")
        prompt = next(
            box for box in dialog.findChildren(QMessageBox) if box.isVisible()
        )
        if change == "busy":
            ctrl.get_device_snapshot.return_value = DeviceSnapshot(
                name="bias",
                type_name="FakeDevice",
                address="none",
                status=DeviceStatus.SETTING_UP,
            )
        elif change == "missing":
            ctrl.get_device_snapshot.return_value = None
        elif change == "read-only":
            dialog.set_read_only(read_only=True)
        else:
            dialog.reject()
        if prompt.isVisible():
            ComponentDriver(prompt).click("Drop")
        ctrl.start_disconnect_device.assert_not_called()
        ctrl.forget_device.assert_not_called()
    finally:
        for box in dialog.findChildren(QMessageBox):
            box.reject()
        dialog.reject()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("field", ["Name:", "Address:"])
def test_enter_in_registration_form_adds_instead_of_dropping_selected_device(
    qapp, field: str, tmp_path: Path
) -> None:
    from zcu_tools.device.fake import FakeDeviceInfo

    ctrl = make_device_control()
    ctrl.list_devices.return_value = [device_entry("bias")]
    ctrl.get_device_snapshot.return_value = connected_snapshot(
        "bias", FakeDeviceInfo(address="none")
    )
    dialog = make_device_dialog(ctrl)
    try:
        dialog.show()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.select_list_row(driver.list_rows()[0].text)
        driver.enter_text("Name:", value="new-bias", commit=False)
        driver.enter_text("Address:", value="none", commit=False)
        driver.enter_text(field, value="new-bias" if field == "Name:" else "none")
        driver.screenshot(tmp_path / f"device-enter-{field[:-1]}.png")
        ctrl.start_connect_device.assert_called_once()
        assert ctrl.start_connect_device.call_args.args[0].name == "new-bias"
        assert not any(box.isVisible() for box in dialog.findChildren(QMessageBox))
        ctrl.start_disconnect_device.assert_not_called()
    finally:
        for box in dialog.findChildren(QMessageBox):
            box.reject()
        dialog.reject()
        dialog.deleteLater()
        qapp.processEvents()
