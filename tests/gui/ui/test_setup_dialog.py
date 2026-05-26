"""Smoke tests for SetupDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.services import StartupConnectionRequest, StartupProjectRequest
from zcu_tools.gui.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
)
from zcu_tools.gui.ui.setup_dialog import SetupDialog


def _make_ctrl(**overrides: object) -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = None
    ctrl.list_devices.return_value = []
    ctrl.get_context_labels.return_value = []
    ctrl.get_active_context_label.return_value = None
    ctrl.get_soccfg.return_value = None
    ctrl.apply_startup_project.return_value = True
    for k, v in overrides.items():
        getattr(ctrl, k).return_value = v
    return ctrl


def test_setup_dialog_init(qapp):
    ctrl = _make_ctrl(
        get_context_labels=["ctx1", "ctx2"],
        get_active_context_label="ctx1",
    )

    dialog = SetupDialog(ctrl)

    assert dialog._ctx_list.count() == 2
    # The first item should be active (index 0)
    assert dialog._ctx_list.currentRow() == 0


def test_setup_dialog_apply_startup_context(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._chip_edit.setText("Q1_Chip")
    dialog._qub_edit.setText("Q1")
    dialog._res_edit.setText("R1")
    dialog._result_dir_edit.setText("/my/result/dir")
    dialog._db_path_edit.setText("/my/db/dir")

    dialog._on_apply_startup_clicked()

    ctrl.apply_startup_project.assert_called_once_with(
        StartupProjectRequest(
            chip_name="Q1_Chip",
            qub_name="Q1",
            res_name="R1",
            result_dir="/my/result/dir",
            database_path="/my/db/dir",
        )
    )


def test_setup_dialog_does_not_render_success_when_project_apply_fails(qapp):
    ctrl = _make_ctrl()
    ctrl.apply_startup_project.return_value = False
    dialog = SetupDialog(ctrl)

    dialog._on_apply_startup_clicked()

    assert "Startup context applied" not in dialog._project_status.text()


def test_setup_dialog_switch_context(qapp):
    ctrl = _make_ctrl(
        get_context_labels=["ctx1", "ctx2"],
        get_active_context_label="ctx1",
    )
    dialog = SetupDialog(ctrl)

    dialog._ctx_list.setCurrentRow(1)  # select ctx2
    dialog._on_switch_clicked()

    ctrl.use_context.assert_called_with("ctx2")


def test_setup_dialog_new_context(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._clone_check.setChecked(True)
    dialog._on_new_ctx_clicked()

    ctrl.new_context.assert_called_with(
        value=None, unit="none", clone_from_current=True
    )


def test_setup_dialog_connect_mock_dispatches_request(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(True)
    dialog._on_connect_clicked()

    ctrl.start_connect.assert_called_once()
    (req,) = ctrl.start_connect.call_args.args
    assert isinstance(req, ConnectMockRequest)


def test_setup_dialog_connect_remote_dispatches_request(qapp):
    ctrl = _make_ctrl()
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
    ctrl.remember_startup_connection.assert_called_once_with(
        StartupConnectionRequest(ip="10.0.0.1", port=7000)
    )


def test_setup_dialog_connect_failure_signal_updates_status(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(True)
    dialog._on_connect_clicked()
    on_failed = ctrl.bind_connection_outcome.call_args.args[1]
    on_failed("network bad")

    assert "network bad" in dialog._conn_status.text()
    assert dialog._connect_btn.isEnabled()
