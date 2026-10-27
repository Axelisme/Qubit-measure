"""SoC dialog connection submission and visible lifecycle contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.services.connection import ConnectRemoteRequest
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    SetupPreferences,
)
from zcu_tools.gui.session.ui.soc_dialog import SocDialog
from zcu_tools.gui.widgets import error_colour

from tests.gui._user_input import ComponentDriver


def _make_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_setup_preferences.return_value = SetupPreferences(
        entry_name="", ip="192.168.10.1", port=8887
    )
    ctrl.get_soccfg.return_value = None
    ctrl.get_bus.return_value = BaseEventBus()
    return ctrl


@pytest.mark.parametrize("field", ["soc_ip", "qt_spinbox_lineedit"])
def test_enter_submits_connection_once(
    qapp: QApplication, field: str, tmp_path: Path
) -> None:
    ctrl = _make_ctrl()
    dialog = SocDialog(ctrl)
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.enter_text(value="10.2.3.4", object_name="soc_ip", commit=False)
        driver.press_key(Qt.Key.Key_Return, object_name=field)
        driver.press_key(Qt.Key.Key_Return, object_name=field)
        driver.screenshot(tmp_path / f"soc-enter-{field}.png")
        ctrl.start_connect.assert_called_once_with(
            ConnectRemoteRequest(ip="10.2.3.4", port=8887)
        )
        assert not driver.observe("Connect").enabled
    finally:
        dialog.reject()
        dialog.deleteLater()
        qapp.processEvents()


def soc_button(dialog: SocDialog, text: str) -> QPushButton:
    return next(
        button for button in dialog.findChildren(QPushButton) if button.text() == text
    )


def test_setup_dialog_connect_remote_dispatches_request(qapp):
    ctrl = _make_ctrl()
    dialog = SocDialog(ctrl)

    connection = next(
        group
        for group in dialog.findChildren(QGroupBox)
        if group.title() == "ZCU Connection"
    )
    ip = connection.findChild(QLineEdit)
    port = connection.findChild(QSpinBox)
    assert ip is not None and port is not None
    ip.setText("10.0.0.1")
    port.setValue(7000)
    soc_button(dialog, "Connect").click()

    ctrl.start_connect.assert_called_once()
    (req,) = ctrl.start_connect.call_args.args
    assert isinstance(req, ConnectRemoteRequest)
    assert req.ip == "10.0.0.1"
    assert req.port == 7000
    ctrl.remember_connection.assert_called_once_with(
        ConnectionPreferences(ip="10.0.0.1", port=7000)
    )


def test_setup_dialog_connect_failure_signal_updates_status(qapp):
    ctrl = _make_ctrl()
    dialog = SocDialog(ctrl)

    connect = soc_button(dialog, "Connect")
    connect.click()
    assert not connect.isEnabled()
    on_failed = ctrl.bind_connection_outcome.call_args.args[1]
    on_failed("network bad")

    assert any("network bad" in label.text() for label in dialog.findChildren(QLabel))
    assert connect.isEnabled()


def test_soc_dialog_read_only_disables_connection_fields_and_keeps_close(
    qapp: QApplication, tmp_path: Path
) -> None:
    ctrl = _make_ctrl()
    dialog = SocDialog(ctrl)
    try:
        dialog.show()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.enter_text("IP address:", value="10.1.2.3", commit=False)
        driver.screenshot(tmp_path / "soc-editable.png")
        assert driver.observe(object_name="soc_ip").text == "10.1.2.3"
        dialog.set_read_only(read_only=True)
        driver.screenshot(tmp_path / "soc-read-only.png")
        assert not driver.observe(object_name="soc_ip").enabled
        assert not driver.observe(object_name="soc_port").enabled
        assert not driver.observe("Connect").enabled
        assert driver.observe("Close").enabled
        driver.click("Close")
        assert not dialog.isVisible()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_soc_dialog_reopen_prefills_preferences_and_observes_soc_changes(
    qapp: QApplication, tmp_path: Path
) -> None:
    ctrl = _make_ctrl()
    bus = ctrl.get_bus()
    dialog = SocDialog(ctrl)
    try:
        dialog.show()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.click("Close")
        ctrl.get_setup_preferences.return_value = SetupPreferences(
            entry_name="", ip="10.4.5.6", port=7000
        )
        dialog.show()
        qapp.processEvents()
        assert driver.observe(object_name="soc_ip").text == "10.4.5.6"
        assert driver.observe(object_name="qt_spinbox_lineedit").text == "7000"
        bus.emit(SocChangedPayload(soc=None, soccfg=None))
        driver.screenshot(tmp_path / "soc-reopened-disconnected.png")
        assert any(widget.text == "Disconnected" for widget in driver.visible().widgets)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_soc_connection_error_uses_theme_foreground(qapp, window_palette):
    ctrl = _make_ctrl()
    ctrl.start_connect.side_effect = RuntimeError("connection failed")
    dialog = SocDialog(ctrl)
    try:
        dialog.setPalette(window_palette)
        soc_button(dialog, "Connect").click()
        label = next(
            label for label in dialog.findChildren(QLabel) if "failed" in label.text()
        )
        label.ensurePolished()
        assert label.palette().color(label.foregroundRole()) == error_colour(label)
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
