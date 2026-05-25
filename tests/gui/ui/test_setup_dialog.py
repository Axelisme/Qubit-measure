"""Smoke tests for SetupDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QWidget  # type: ignore[attr-defined]
from zcu_tools.gui.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
)
from zcu_tools.gui.ui.setup_dialog import SetupDialog


class _StubConnSvc(QWidget):
    """Minimal Qt object exposing the two ConnectionService signals."""

    connection_finished: Signal = Signal()
    connection_failed: Signal = Signal(str)


def test_setup_dialog_init(qapp):
    ctrl = MagicMock()
    ctrl.get_context_labels.return_value = ["ctx1", "ctx2"]
    ctrl.get_active_context_label.return_value = "ctx1"
    ctrl.get_soccfg.return_value = None

    dialog = SetupDialog(ctrl)

    assert dialog._ctx_list.count() == 2
    # The first item should be active (index 0)
    assert dialog._ctx_list.currentRow() == 0


def test_setup_dialog_apply_startup_context(qapp):
    ctrl = MagicMock()
    ctrl.get_soccfg.return_value.description.return_value = "Mock SOC config"
    dialog = SetupDialog(ctrl)

    dialog._chip_edit.setText("Q1_Chip")
    dialog._qub_edit.setText("Q1")
    dialog._res_edit.setText("R1")
    dialog._result_dir_edit.setText("/my/result/dir")
    dialog._db_path_edit.setText("/my/db/dir")

    dialog._on_apply_startup_clicked()

    ctrl.set_startup_context.assert_called_once()
    ctrl.setup_project.assert_called_with("/my/result/dir")


def test_setup_dialog_switch_context(qapp):
    ctrl = MagicMock()
    ctrl.get_context_labels.return_value = ["ctx1", "ctx2"]
    ctrl.get_active_context_label.return_value = "ctx1"
    ctrl.get_soccfg.return_value.description.return_value = "Mock SOC config"
    dialog = SetupDialog(ctrl)

    dialog._ctx_list.setCurrentRow(1)  # select ctx2
    dialog._on_switch_clicked()

    ctrl.use_context.assert_called_with("ctx2")


def test_setup_dialog_new_context(qapp):
    ctrl = MagicMock()
    ctrl.get_soccfg.return_value.description.return_value = "Mock SOC config"
    dialog = SetupDialog(ctrl)

    dialog._clone_check.setChecked(True)
    dialog._on_new_ctx_clicked()

    ctrl.new_context.assert_called_with(
        value=None, unit="none", clone_from_current=True
    )


def test_setup_dialog_connect_mock_dispatches_request(qapp):
    ctrl = MagicMock()
    ctrl.get_soccfg.return_value.description.return_value = "Mock SOC config"
    stub_conn = _StubConnSvc()
    ctrl.get_connection_service.return_value = stub_conn
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(True)
    dialog._on_connect_clicked()

    ctrl.start_connect.assert_called_once()
    (req,) = ctrl.start_connect.call_args.args
    assert isinstance(req, ConnectMockRequest)


def test_setup_dialog_connect_remote_dispatches_request(qapp):
    ctrl = MagicMock()
    ctrl.get_soccfg.return_value.description.return_value = "Mock SOC config"
    stub_conn = _StubConnSvc()
    ctrl.get_connection_service.return_value = stub_conn
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(False)
    dialog._ip_edit.setText("10.0.0.1")
    dialog._port_spin.setValue(7000)
    dialog._on_connect_clicked()

    ctrl.start_connect.assert_called_once()
    (req,) = ctrl.start_connect.call_args.args
    assert isinstance(req, ConnectRemoteRequest)
    assert req.ip == "10.0.0.1"
    assert req.port == 7000


def test_setup_dialog_connect_failure_signal_updates_status(qapp):
    ctrl = MagicMock()
    ctrl.get_soccfg.return_value.description.return_value = "Mock SOC config"
    stub_conn = _StubConnSvc()
    ctrl.get_connection_service.return_value = stub_conn
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(True)
    dialog._on_connect_clicked()
    stub_conn.connection_failed.emit("network bad")

    assert "network bad" in dialog._conn_status.text()
    assert dialog._connect_btn.isEnabled()
