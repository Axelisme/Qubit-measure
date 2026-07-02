"""Smoke tests for SetupDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFormLayout,
    QGroupBox,
    QLabel,
    QWidget,
)
from zcu_tools.gui.result_scope import ResultScope, ResultScopeManager
from zcu_tools.gui.session.services.connection import (
    ConnectMockRequest,
    ConnectRemoteRequest,
)
from zcu_tools.gui.session.services.startup import (
    PersistedStartup,
    StartupConnectionRequest,
    StartupProjectRequest,
)
from zcu_tools.gui.session.ui.setup_dialog import SetupDialog


def _make_ctrl(**overrides: object) -> MagicMock:
    ctrl = MagicMock()
    # SetupControlPort.get_persisted_startup is non-Optional (a default
    # PersistedStartup when nothing is remembered), so the double honours that
    # contract rather than returning None.
    ctrl.get_persisted_startup.return_value = PersistedStartup()
    ctrl.list_devices.return_value = []
    ctrl.get_context_labels.return_value = []
    ctrl.get_active_context_label.return_value = None
    ctrl.get_soccfg.return_value = None
    ctrl.apply_startup_project.return_value = True
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


def test_setup_dialog_apply_startup_context(qapp):
    ctrl = _make_ctrl()
    dialog = SetupDialog(ctrl)

    dialog._chip_edit.setText("Q1_Chip")
    dialog._qub_edit.setText("Q1")
    dialog._res_edit.setText("R1")

    dialog._on_apply_startup_clicked()

    ctrl.apply_startup_project.assert_called_once_with(
        StartupProjectRequest(
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


def test_setup_dialog_reseed_on_reshow_clears_stale_draft(qapp):
    """Regression: re-raising a dialog with an un-applied draft must reset to
    the current State (startup_prefs), not retain the typed-but-not-applied value.

    Scenario: open dialog (shows chip="Q5_2D") → user types "DRAFT" without
    applying → dialog is re-shown (simulated by calling showEvent directly, as
    open_dialog does raise_()+show()) → chip field reverts to "Q5_2D".
    """
    prefs = PersistedStartup(chip_name="Q5_2D", qub_name="Q1", res_name="R1")
    ctrl = _make_ctrl(get_persisted_startup=prefs)

    dialog = SetupDialog(ctrl)
    # after init: chip_edit should reflect the persisted prefs
    assert dialog._chip_edit.text() == "Q5_2D"

    # user types a draft — no apply
    dialog._chip_edit.setText("DRAFT_NAME")
    assert dialog._chip_edit.text() == "DRAFT_NAME"

    # simulate the dialog being re-raised (open_dialog calls raise_()/show();
    # showEvent fires on every show call)
    dialog.showEvent(None)

    # the re-seed must revert to the State value, not the stale draft
    assert dialog._chip_edit.text() == "Q5_2D"
