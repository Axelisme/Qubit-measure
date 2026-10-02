"""Smoke tests for SetupDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QWidget,
)
from zcu_tools.gui.result_scope import ResultScope, ResultScopeManager
from zcu_tools.gui.session.services.connection import (
    ConnectRemoteRequest,
)
from zcu_tools.gui.session.services.project_settings import (
    ConnectionPreferences,
    ProjectRequest,
    SetupPreferences,
)
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog


def _prefs(
    *,
    chip_name: str = "",
    qub_name: str = "",
    res_name: str = "",
    scope_id: str = "",
    ip: str = "192.168.10.1",
    port: int = 8887,
) -> SetupPreferences:
    return SetupPreferences(
        chip_name=chip_name,
        qub_name=qub_name,
        res_name=res_name,
        scope_id=scope_id,
        ip=ip,
        port=port,
    )


def _make_ctrl(**overrides: object) -> MagicMock:
    ctrl = MagicMock()
    # SetupControlPort.get_setup_preferences is non-Optional (defaults when
    # nothing is remembered), so the double honours that contract rather than
    # returning None.
    ctrl.get_setup_preferences.return_value = _prefs()
    ctrl.list_devices.return_value = []
    ctrl.get_context_labels.return_value = []
    ctrl.get_active_context_label.return_value = None
    ctrl.get_soccfg.return_value = None
    ctrl.apply_project.return_value = True
    ctrl.get_project_root.return_value = "/tmp"
    ctrl.list_result_scopes.return_value = ()
    manager = ResultScopeManager("/tmp")
    ctrl.derive_project_paths.side_effect = manager.derive_paths
    for k, v in overrides.items():
        getattr(ctrl, k).return_value = v
    return ctrl


def _nearest_group(widget: QWidget) -> QGroupBox | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QGroupBox):
            return parent
        parent = parent.parentWidget()
    return None


def _form_label_rows(form: QFormLayout) -> dict[str, int]:
    rows: dict[str, int] = {}
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        if item is None:
            continue
        widget = item.widget()
        if not isinstance(widget, QLabel):
            continue
        rows[widget.text()] = row
    return rows


def test_setup_dialog_init(qapp):
    ctrl = _make_ctrl(
        get_context_labels=["ctx1", "ctx2"],
        get_active_context_label="ctx1",
    )

    dialog = SetupDialog(ctrl)

    assert dialog._ctx_list.count() == 2
    # The first item should be active (index 0)
    assert dialog._ctx_list.currentRow() == 0


def test_setup_dialog_project_scope_names_and_apply_share_group(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    project_group = _nearest_group(dialog._scope_combo)
    assert project_group is not None
    for widget in (
        dialog._chip_edit,
        dialog._qub_edit,
        dialog._res_edit,
        dialog._apply_btn,
    ):
        assert _nearest_group(widget) is project_group

    form = project_group.layout()
    assert isinstance(form, QFormLayout)
    rows = _form_label_rows(form)
    assert rows["Scope:"] < rows["Chip name:"]
    assert rows["Chip name:"] < rows["Qubit name:"]
    assert rows["Qubit name:"] < rows["Resonator name:"]
    assert "Result dir:" not in rows
    assert "Database path:" not in rows


def test_setup_dialog_apply_project(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._chip_edit.setText("Q1_Chip")
    dialog._qub_edit.setText("Q1")
    dialog._res_edit.setText("R1")

    dialog._on_apply_clicked()

    ctrl.apply_project.assert_called_once_with(
        ProjectRequest(
            chip_name="Q1_Chip",
            qub_name="Q1",
            res_name="R1",
        )
    )


def test_setup_dialog_scope_combo_lists_all_scopes_and_selection_prefills_names(qapp):
    scope = ResultScope(
        scope_id="/tmp/result/Q3_2D/Q1",
        chip_name="Q3_2D",
        qub_name="Q1",
        result_dir="/tmp/result/Q3_2D/Q1",
        params_path="/tmp/result/Q3_2D/Q1/params.json",
        source="discovered",
    )
    ctrl = _make_ctrl(list_result_scopes=(scope,))

    dialog = SetupDialog(ctrl)

    idx = dialog._scope_combo.findData(scope.scope_id)
    assert idx >= 0
    assert dialog._scope_combo.itemText(idx) == "Q3_2D/Q1"
    dialog._scope_combo.setCurrentIndex(idx)

    assert dialog._chip_edit.text() == "Q3_2D"
    assert dialog._qub_edit.text() == "Q1"


def test_setup_dialog_prefills_persisted_scope_id_without_side_effects(qapp):
    scope = ResultScope(
        scope_id="/tmp/result/Q4_2D/Q2",
        chip_name="Q4_2D",
        qub_name="Q2",
        result_dir="/tmp/result/Q4_2D/Q2",
        params_path="/tmp/result/Q4_2D/Q2/params.json",
        source="discovered",
    )
    prefs = _prefs(
        chip_name="Q4_2D",
        qub_name="Q2",
        res_name="R2",
        scope_id=scope.scope_id,
        ip="10.0.0.2",
        port=7777,
    )
    ctrl = _make_ctrl(get_setup_preferences=prefs, list_result_scopes=(scope,))

    dialog = SetupDialog(ctrl)

    assert dialog._scope_combo.currentData() == scope.scope_id
    assert dialog._chip_edit.text() == "Q4_2D"
    assert dialog._qub_edit.text() == "Q2"
    assert dialog._res_edit.text() == "R2"
    assert dialog._ip_edit.text() == "10.0.0.2"
    assert dialog._port_spin.value() == 7777
    ctrl.apply_project.assert_not_called()
    ctrl.remember_connection.assert_not_called()
    ctrl.start_connect.assert_not_called()


def test_setup_dialog_missing_persisted_scope_falls_back_without_side_effects(qapp):
    scope = ResultScope(
        scope_id="/tmp/result/Q5_2D/Q1",
        chip_name="Q5_2D",
        qub_name="Q1",
        result_dir="/tmp/result/Q5_2D/Q1",
        params_path="/tmp/result/Q5_2D/Q1/params.json",
        source="discovered",
    )
    prefs = _prefs(
        chip_name="Q5_2D",
        qub_name="Q1",
        res_name="R1",
        scope_id="/tmp/result/missing/scope",
    )
    ctrl = _make_ctrl(get_setup_preferences=prefs, list_result_scopes=(scope,))

    dialog = SetupDialog(ctrl)

    assert dialog._scope_combo.currentData() == scope.scope_id
    assert dialog._chip_edit.text() == "Q5_2D"
    assert dialog._qub_edit.text() == "Q1"
    ctrl.apply_project.assert_not_called()
    ctrl.remember_connection.assert_not_called()
    ctrl.start_connect.assert_not_called()


def test_setup_dialog_does_not_render_success_when_project_apply_fails(qapp):
    ctrl = _make_ctrl()
    ctrl.apply_project.return_value = False
    dialog = SetupDialog(ctrl)

    dialog._on_apply_clicked()

    assert "Project applied" not in dialog._project_status.text()


def test_setup_dialog_switch_context(qapp):
    ctrl = _make_ctrl(
        get_context_labels=["ctx1", "ctx2"],
        get_active_context_label="ctx1",
    )
    dialog = SetupDialog(ctrl)

    dialog._ctx_list.setCurrentRow(1)  # select ctx2
    dialog._on_switch_clicked()

    ctrl.use_context.assert_called_with("ctx2")


def test_setup_dialog_new_context_clone_from_dropdown(qapp):
    # No bound device (empty device combo) -> bind_device=None; the clone
    # dropdown is populated from the active project's context labels and the
    # picked label flows through as clone_from.
    ctrl = _make_ctrl(
        get_context_labels=["ctx_a", "ctx_b"],
        get_active_context_label="ctx_a",
    )
    dialog = SetupDialog(ctrl)

    # index 0 == "(none)"; pick "ctx_b".
    idx = dialog._clone_combo.findData("ctx_b")
    assert idx > 0
    dialog._clone_combo.setCurrentIndex(idx)
    dialog._on_new_ctx_clicked()

    ctrl.new_context.assert_called_with(bind_device=None, clone_from="ctx_b")


def test_setup_dialog_new_context_clone_none_default(qapp):
    # Default clone selection "(none)" -> clone_from=None.
    ctrl = _make_ctrl(get_context_labels=["ctx_a"])
    dialog = SetupDialog(ctrl)

    dialog._on_new_ctx_clicked()

    ctrl.new_context.assert_called_with(bind_device=None, clone_from=None)


def test_setup_dialog_connect_mock_dispatches_request(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._mock_check.setChecked(True)
    next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Connect"
    ).click()

    ctrl.start_simulated_environment.assert_called_once_with()


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
    ctrl.remember_connection.assert_called_once_with(
        ConnectionPreferences(ip="10.0.0.1", port=7000)
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


def test_setup_dialog_reseed_on_reshow_clears_stale_draft(qapp):
    """Regression: re-raising a dialog with an un-applied draft must reset to
    the current State (preferences), not retain the typed-but-not-applied value.

    Scenario: open dialog (shows chip="Q5_2D") → user types "DRAFT" without
    applying → dialog is re-shown (simulated by calling showEvent directly, as
    open_dialog does raise_()+show()) → chip field reverts to "Q5_2D".
    """
    prefs = _prefs(chip_name="Q5_2D", qub_name="Q1", res_name="R1")
    ctrl = _make_ctrl(get_setup_preferences=prefs)

    dialog = SetupDialog(ctrl)
    # after init: chip_edit should reflect the remembered preferences
    assert dialog._chip_edit.text() == "Q5_2D"

    # user types a draft — no apply
    dialog._chip_edit.setText("DRAFT_NAME")
    assert dialog._chip_edit.text() == "DRAFT_NAME"

    # simulate the dialog being re-raised (open_dialog calls raise_()/show();
    # showEvent fires on every show call)
    dialog.showEvent(None)

    # the re-seed must revert to the State value, not the stale draft
    assert dialog._chip_edit.text() == "Q5_2D"
