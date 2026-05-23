"""Smoke tests for SetupDialog."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from zcu_tools.gui.ui.setup_dialog import SetupDialog


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
    dialog = SetupDialog(ctrl)

    dialog._ctx_list.setCurrentRow(1)  # select ctx2
    dialog._on_switch_clicked()

    ctrl.use_context.assert_called_with("ctx2")


def test_setup_dialog_new_context(qapp):
    ctrl = MagicMock()
    dialog = SetupDialog(ctrl)

    dialog._clone_check.setChecked(True)
    dialog._on_new_ctx_clicked()

    ctrl.new_context.assert_called_with(
        value=None, unit="none", clone_from_current=True
    )


def test_setup_dialog_connect_mock(qapp):
    ctrl = MagicMock()
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(True)

    with (
        patch("zcu_tools.program.v2.mocksoc.make_mock_soc") as make_soc,
        patch("zcu_tools.program.v2.mocksoc.make_mock_soccfg") as make_cfg,
    ):
        mock_soc = MagicMock()
        mock_cfg = MagicMock()
        make_soc.return_value = mock_soc
        make_cfg.return_value = mock_cfg

        dialog._on_connect_clicked()

        ctrl.set_connection.assert_called_with(mock_soc, mock_cfg)
