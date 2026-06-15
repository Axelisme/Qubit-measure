"""Tests for PredictorDialog — tracked-transitions table UX."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from zcu_tools.gui.session.services.predictor import (
    PredictCurveResult,
    PredictMatrixCurveResult,
    PredictorLoadError,
    PredictorNotLoaded,
    SetModelParamsRequest,
)
from zcu_tools.gui.session.ui.predictor_dialog import (
    _COL_FREQ,
    _COL_MAG,
    _DEFAULT_EC,
    _DEFAULT_EJ,
    _DEFAULT_EL,
    _DEFAULT_FLUX_BIAS,
    _DEFAULT_FLUX_HALF,
    _DEFAULT_FLUX_PERIOD,
    _DEFAULT_TRANSITIONS,
    PredictorDialog,
)


def _make_freq_result(
    transitions: list[tuple[int, int]], n: int = 20
) -> PredictCurveResult:
    """Build a minimal PredictCurveResult for the given transitions."""
    values = np.linspace(-0.5, 0.5, n)
    labels = tuple(f"{f}→{t}" for f, t in transitions)
    freqs = np.ones((len(transitions), n), dtype=np.float64) * 1000.0
    return PredictCurveResult(
        labels=labels,
        values=values,
        fluxs=values.copy(),
        freqs_mhz=freqs,
    )


def _make_mat_result(
    transitions: list[tuple[int, int]], n: int = 20
) -> PredictMatrixCurveResult:
    """Build a minimal PredictMatrixCurveResult for the given transitions."""
    values = np.linspace(-0.5, 0.5, n)
    labels = tuple(f"{f}→{t}" for f, t in transitions)
    mags = np.ones((len(transitions), n), dtype=np.float64) * 0.3
    return PredictMatrixCurveResult(
        labels=labels,
        values=values,
        fluxs=values.copy(),
        mags=mags,
    )


def _make_ctrl(
    *,
    has_predictor: bool = False,
    flux_half: float = 0.5,
    flux_period: float = 1.0,
    flux_bias: float = 0.0,
    ej: float = 4.0,
    ec: float = 1.0,
    el: float = 1.0,
    path: str | None = None,
) -> MagicMock:
    """Build a MagicMock controller pre-configured for dialog tests."""
    ctrl = MagicMock()
    if has_predictor:
        ctrl.get_predictor_info.return_value = {
            "path": path,
            "flux_bias": flux_bias,
            "flux_half": flux_half,
            "flux_period": flux_period,
            "EJ": ej,
            "EC": ec,
            "EL": el,
        }
    else:
        ctrl.get_predictor_info.return_value = None

    def _freq_side(req):  # type: ignore[no-untyped-def]
        return _make_freq_result(list(req.transitions))

    def _mat_side(req):  # type: ignore[no-untyped-def]
        return _make_mat_result(list(req.transitions))

    ctrl.predict_freq_curve.side_effect = _freq_side
    ctrl.predict_matrix_element_curve.side_effect = _mat_side
    ctrl.predict_freq.return_value = 1234.5
    return ctrl


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------


def test_predictor_dialog_default_field_values_no_predictor(qapp):
    """With no predictor loaded, the six spinboxes show the documented defaults."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    assert dialog._ej_spin.value() == pytest.approx(_DEFAULT_EJ)  # 4.0
    assert dialog._ec_spin.value() == pytest.approx(_DEFAULT_EC)  # 1.0
    assert dialog._el_spin.value() == pytest.approx(_DEFAULT_EL)  # 1.0
    assert dialog._flux_half_spin.value() == pytest.approx(_DEFAULT_FLUX_HALF)  # 0.0
    assert dialog._flux_period_spin.value() == pytest.approx(
        _DEFAULT_FLUX_PERIOD
    )  # 0.005
    assert dialog._flux_bias_spin.value() == pytest.approx(_DEFAULT_FLUX_BIAS)  # 0.0


# ---------------------------------------------------------------------------
# Init / load
# ---------------------------------------------------------------------------


def test_predictor_dialog_init_populates_fields_from_info(qapp):
    """With a predictor loaded at init, the six spinboxes reflect it."""
    ctrl = _make_ctrl(
        has_predictor=True,
        path="/p.json",
        ej=4.2,
        ec=1.1,
        el=0.7,
        flux_half=0.3,
        flux_period=0.8,
        flux_bias=0.05,
    )
    dialog = PredictorDialog(ctrl)

    assert dialog._ej_spin.value() == pytest.approx(4.2)
    assert dialog._ec_spin.value() == pytest.approx(1.1)
    assert dialog._el_spin.value() == pytest.approx(0.7)
    assert dialog._flux_half_spin.value() == pytest.approx(0.3)
    assert dialog._flux_period_spin.value() == pytest.approx(0.8)
    assert dialog._flux_bias_spin.value() == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Browse button — populates six fields, no install
# ---------------------------------------------------------------------------


def test_predictor_dialog_browse_populates_six_fields(qapp):
    """Browse picks a file and populates EJ/EC/EL/flux_half/flux_period spinboxes."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    fake_params = SetModelParamsRequest(
        EJ=5.5, EC=1.3, EL=0.6, flux_half=0.4, flux_period=0.7
    )
    with (
        patch(
            "zcu_tools.gui.session.ui.predictor_dialog.QFileDialog.getOpenFileName",
            return_value=("/chosen/params.json", ""),
        ),
        patch(
            "zcu_tools.gui.session.services.predictor.read_fluxdep_fit_params",
            return_value=fake_params,
        ),
    ):
        dialog._on_browse_file()

    assert dialog._ej_spin.value() == pytest.approx(5.5)
    assert dialog._ec_spin.value() == pytest.approx(1.3)
    assert dialog._el_spin.value() == pytest.approx(0.6)
    assert dialog._flux_half_spin.value() == pytest.approx(0.4)
    assert dialog._flux_period_spin.value() == pytest.approx(0.7)
    # Browse must NOT install a predictor.
    ctrl.set_predictor_model_params.assert_not_called()


def test_predictor_dialog_browse_cancel_is_noop(qapp):
    """Cancelling the file dialog (empty path) leaves the fields unchanged."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    before_ej = dialog._ej_spin.value()

    with patch(
        "zcu_tools.gui.session.ui.predictor_dialog.QFileDialog.getOpenFileName",
        return_value=("", ""),
    ):
        dialog._on_browse_file()

    assert dialog._ej_spin.value() == before_ej
    ctrl.set_predictor_model_params.assert_not_called()


def test_predictor_dialog_browse_error_shown_in_status(qapp):
    """A read error during Browse is surfaced in the status label, fields untouched."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    with (
        patch(
            "zcu_tools.gui.session.ui.predictor_dialog.QFileDialog.getOpenFileName",
            return_value=("/bad/params.json", ""),
        ),
        patch(
            "zcu_tools.gui.session.services.predictor.read_fluxdep_fit_params",
            side_effect=PredictorLoadError("no fluxdep_fit"),
        ),
    ):
        dialog._on_browse_file()

    assert "no fluxdep_fit" in dialog._status_label.text()


# ---------------------------------------------------------------------------
# Apply button
# ---------------------------------------------------------------------------


def test_predictor_dialog_apply_installs_from_fields(qapp):
    """Apply builds a SetModelParamsRequest from the fields and installs it."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._ej_spin.setValue(4.0)
    dialog._ec_spin.setValue(1.0)
    dialog._el_spin.setValue(1.0)
    dialog._flux_half_spin.setValue(0.25)
    dialog._flux_period_spin.setValue(0.9)
    dialog._flux_bias_spin.setValue(0.05)

    dialog._on_apply_model_params()

    ctrl.set_predictor_model_params.assert_called_once()
    (req,) = ctrl.set_predictor_model_params.call_args.args
    assert isinstance(req, SetModelParamsRequest)
    assert req.EJ == pytest.approx(4.0)
    assert req.EC == pytest.approx(1.0)
    assert req.EL == pytest.approx(1.0)
    assert req.flux_half == pytest.approx(0.25)
    assert req.flux_period == pytest.approx(0.9)
    assert req.flux_bias == pytest.approx(0.05)


def test_predictor_dialog_apply_zero_period_guarded(qapp):
    """flux_period == 0 must NOT install — the dialog fast-fails with a message."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._flux_period_spin.setValue(0.0)
    dialog._on_apply_model_params()

    ctrl.set_predictor_model_params.assert_not_called()
    assert "flux_period" in dialog._status_label.text()


def test_predictor_dialog_apply_surfaces_service_error(qapp):
    """A PredictorLoadError from the controller is shown in the status label."""
    ctrl = _make_ctrl(has_predictor=False)
    ctrl.set_predictor_model_params.side_effect = PredictorLoadError("bad model")
    dialog = PredictorDialog(ctrl)

    dialog._flux_period_spin.setValue(1.0)
    dialog._on_apply_model_params()

    assert "bad model" in dialog._status_label.text()


# ---------------------------------------------------------------------------
# Default tracked transitions — table layout
# ---------------------------------------------------------------------------


def test_predictor_dialog_default_tracked_count(qapp):
    """Dialog initialises with 5 default tracked transitions, table has 5 rows."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert len(dialog._tracked) == 5
    assert dialog._table.rowCount() == 5


def test_predictor_dialog_table_has_3_columns(qapp):
    """Table must have 3 columns: Transition | f (MHz) | |M|."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert dialog._table.columnCount() == 3


def test_predictor_dialog_right_side_has_two_tabs(qapp):
    """Right panel is a QTabWidget with exactly 2 tabs: Frequency and Matrix element."""
    from qtpy.QtWidgets import QTabWidget  # type: ignore[attr-defined]

    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert isinstance(dialog._tab_widget, QTabWidget)
    assert dialog._tab_widget.count() == 2
    assert dialog._tab_widget.tabText(0) == "Frequency"
    assert dialog._tab_widget.tabText(1) == "Matrix element"


def test_predictor_dialog_default_predict_freq_curve_call(qapp):
    """With predictor loaded at init, predict_freq_curve is called with the 5 defaults."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    PredictorDialog(ctrl)
    ctrl.predict_freq_curve.assert_called()
    last_req = ctrl.predict_freq_curve.call_args.args[0]
    assert set(last_req.transitions) == set(map(tuple, _DEFAULT_TRANSITIONS))


def test_predictor_dialog_default_predict_matrix_element_curve_call(qapp):
    """With predictor loaded at init, predict_matrix_element_curve is called.

    _refresh_curves calls it once with all transitions; _update_value_columns then
    calls it once per transition (single-point column fill).  We verify at least one
    call used the full default transition set AND all calls used operator "n".
    """
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    PredictorDialog(ctrl)
    ctrl.predict_matrix_element_curve.assert_called()
    all_reqs = [
        call.args[0] for call in ctrl.predict_matrix_element_curve.call_args_list
    ]
    # At least one call covers all 5 default transitions (the _refresh_curves call).
    full_call = next(
        (
            r
            for r in all_reqs
            if set(r.transitions) == set(map(tuple, _DEFAULT_TRANSITIONS))
        ),
        None,
    )
    assert full_call is not None, "No call with full default transition set found"
    # All calls must use the default operator "n".
    assert all(r.operator == "n" for r in all_reqs)


def test_predictor_dialog_init_no_predictor_no_curve_call(qapp):
    """With no predictor at init, neither curve function is called."""
    ctrl = _make_ctrl(has_predictor=False)
    PredictorDialog(ctrl)
    ctrl.predict_freq_curve.assert_not_called()
    ctrl.predict_matrix_element_curve.assert_not_called()


def test_predictor_dialog_init_with_predictor_both_canvases_have_axes(qapp):
    """After init with a loaded predictor, both canvases have axes (curves were drawn)."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    assert len(dialog._freq_canvas.figure.get_axes()) > 0
    assert len(dialog._mat_canvas.figure.get_axes()) > 0


# ---------------------------------------------------------------------------
# Add transition
# ---------------------------------------------------------------------------


def test_predictor_dialog_add_valid_transition(qapp):
    """Adding a valid, non-duplicate transition increments row count and calls refresh."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    dialog._add_from_spin.setValue(2)
    dialog._add_to_spin.setValue(4)
    dialog._on_add_clicked()

    assert len(dialog._tracked) == 6
    assert dialog._table.rowCount() == 6
    assert (2, 4) in dialog._tracked
    assert ctrl.predict_freq_curve.call_count > initial_count


def test_predictor_dialog_add_invalid_from_ge_to(qapp):
    """from >= to: not added, row count unchanged, no crash."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._add_from_spin.setValue(3)
    dialog._add_to_spin.setValue(2)
    dialog._on_add_clicked()

    assert len(dialog._tracked) == 5
    assert dialog._table.rowCount() == 5


def test_predictor_dialog_add_duplicate_transition(qapp):
    """Adding a duplicate (0,1): not added, row count unchanged."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)

    dialog._add_from_spin.setValue(0)
    dialog._add_to_spin.setValue(1)
    dialog._on_add_clicked()

    assert len(dialog._tracked) == 5
    assert dialog._table.rowCount() == 5


# ---------------------------------------------------------------------------
# Delete transition (via _delete_transition internal API — still used internally)
# ---------------------------------------------------------------------------


def test_predictor_dialog_delete_transition(qapp):
    """Deleting a tracked transition removes it and triggers a curve refresh."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    transition_to_delete = dialog._tracked[0]
    dialog._delete_transition(transition_to_delete)

    assert len(dialog._tracked) == 4
    assert dialog._table.rowCount() == 4
    assert transition_to_delete not in dialog._tracked
    assert ctrl.predict_freq_curve.call_count > initial_count


def test_predictor_dialog_delete_all_transitions_clears_both_canvases(qapp):
    """Deleting all tracked transitions results in an empty table and cleared canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    for t in list(dialog._tracked):
        dialog._delete_transition(t)

    assert dialog._table.rowCount() == 0
    # Both canvases clear() resets marker_value.
    assert dialog._freq_canvas._marker_value is None
    assert dialog._mat_canvas._marker_value is None


# ---------------------------------------------------------------------------
# Multi-select Remove button
# ---------------------------------------------------------------------------


def test_predictor_dialog_remove_selected_single_row(qapp):
    """Selecting one row and clicking Remove deletes that transition."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    target = dialog._tracked[1]  # (0, 2)
    dialog._table.selectRow(1)
    dialog._on_remove_selected()

    assert target not in dialog._tracked
    assert len(dialog._tracked) == 4
    assert dialog._table.rowCount() == 4
    assert ctrl.predict_freq_curve.call_count > initial_count


def test_predictor_dialog_remove_selected_multiple_rows(qapp):
    """Selecting multiple rows and clicking Remove deletes all selected transitions."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    # Select rows 0 and 2.
    targets = [dialog._tracked[0], dialog._tracked[2]]
    dialog._table.selectRow(0)
    # Extend selection to row 2 via selectRow on the QTableWidget
    # (selectRow replaces selection; use setRangeSelected for multi-select).
    from qtpy.QtWidgets import QTableWidgetSelectionRange  # type: ignore[attr-defined]

    dialog._table.setRangeSelected(
        QTableWidgetSelectionRange(0, 0, 0, dialog._table.columnCount() - 1), True
    )
    dialog._table.setRangeSelected(
        QTableWidgetSelectionRange(2, 0, 2, dialog._table.columnCount() - 1), True
    )
    dialog._on_remove_selected()

    for t in targets:
        assert t not in dialog._tracked
    assert len(dialog._tracked) == 3
    assert dialog._table.rowCount() == 3


def test_predictor_dialog_remove_no_selection_is_noop(qapp):
    """Clicking Remove with nothing selected does nothing."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_freq_count = ctrl.predict_freq_curve.call_count

    dialog._table.clearSelection()
    dialog._on_remove_selected()

    assert len(dialog._tracked) == 5
    assert ctrl.predict_freq_curve.call_count == initial_freq_count


def test_predictor_dialog_remove_all_clears_canvases(qapp):
    """Removing all transitions via multi-select Remove clears both canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    from qtpy.QtWidgets import QTableWidgetSelectionRange  # type: ignore[attr-defined]

    n = dialog._table.rowCount()
    dialog._table.setRangeSelected(
        QTableWidgetSelectionRange(0, 0, n - 1, dialog._table.columnCount() - 1), True
    )
    dialog._on_remove_selected()

    assert dialog._table.rowCount() == 0
    assert dialog._freq_canvas._marker_value is None
    assert dialog._mat_canvas._marker_value is None


# ---------------------------------------------------------------------------
# Operator change
# ---------------------------------------------------------------------------


def test_predictor_dialog_operator_change_triggers_refresh(qapp):
    """Changing the operator combobox re-calls predict_matrix_element_curve."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_mat_count = ctrl.predict_matrix_element_curve.call_count

    dialog._operator_combo.setCurrentText("phi")

    # Changing operator triggers _on_operator_changed → _refresh_curves
    assert ctrl.predict_matrix_element_curve.call_count > initial_mat_count
    last_req = ctrl.predict_matrix_element_curve.call_args.args[0]
    assert last_req.operator == "phi"


def test_predictor_dialog_operator_stored_in_self(qapp):
    """_operator attribute reflects combobox selection."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert dialog._operator == "n"
    # Simulate operator change without predictor — no crash.
    dialog._operator_combo.setCurrentText("phi")
    assert dialog._operator == "phi"


# ---------------------------------------------------------------------------
# Spinbox / debounce → column update
# ---------------------------------------------------------------------------


def test_predictor_dialog_spinbox_change_does_not_recompute_curves(qapp):
    """Changing the Flux value spinbox must NOT call predict_freq_curve again."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    dialog._on_spinbox_changed(0.3)

    assert ctrl.predict_freq_curve.call_count == initial_count


def test_predictor_dialog_spinbox_change_updates_both_markers(qapp):
    """Changing spinbox calls set_marker on both canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    with (
        patch.object(dialog._freq_canvas, "set_marker") as freq_mock,
        patch.object(dialog._mat_canvas, "set_marker") as mat_mock,
    ):
        dialog._on_spinbox_changed(0.3)

    freq_mock.assert_called_with(0.3)
    mat_mock.assert_called_with(0.3)


def test_predictor_dialog_value_columns_updated_after_debounce(qapp):
    """After _update_value_columns, each tracked row's Freq and |M| cells are filled."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    dialog._predict_value_spin.setValue(0.1)
    dialog._update_value_columns()

    # Each row should now show a numeric freq (not "—").
    for row in range(dialog._table.rowCount()):
        freq_item = dialog._table.item(row, _COL_FREQ)
        assert freq_item is not None
        assert freq_item.text() != "—"
        # |M| should also be filled.
        mag_item = dialog._table.item(row, _COL_MAG)
        assert mag_item is not None
        assert mag_item.text() != "—"


def test_predictor_dialog_columns_dash_when_not_loaded(qapp):
    """predict_freq raising PredictorNotLoaded → both columns show '—', no crash."""
    ctrl = _make_ctrl(has_predictor=False)
    ctrl.predict_freq.side_effect = PredictorNotLoaded("no predictor")
    ctrl.predict_matrix_element_curve.side_effect = PredictorNotLoaded("no predictor")
    dialog = PredictorDialog(ctrl)

    dialog._update_value_columns()

    for row in range(dialog._table.rowCount()):
        freq_item = dialog._table.item(row, _COL_FREQ)
        assert freq_item is not None and freq_item.text() == "—"
        mag_item = dialog._table.item(row, _COL_MAG)
        assert mag_item is not None and mag_item.text() == "—"


def test_predictor_dialog_canvas_drag_updates_spinbox_no_loop(qapp):
    """Canvas on_drag must update the spinbox without triggering infinite recursion."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    freq_curve_count_before = ctrl.predict_freq_curve.call_count

    dialog._on_canvas_drag(0.77)

    assert dialog._predict_value_spin.value() == pytest.approx(0.77)
    # on_drag must NOT trigger a full curve recompute.
    assert ctrl.predict_freq_curve.call_count == freq_curve_count_before


# ---------------------------------------------------------------------------
# Table selection → both canvas highlights
# ---------------------------------------------------------------------------


def test_predictor_dialog_row_select_calls_set_highlight_both_canvases(qapp):
    """Selecting a table row calls set_highlight on BOTH canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    with (
        patch.object(dialog._freq_canvas, "set_highlight") as freq_hi,
        patch.object(dialog._mat_canvas, "set_highlight") as mat_hi,
    ):
        dialog._table.selectRow(1)
        dialog._on_selection_changed()

    expected = dialog._tracked[1]
    freq_hi.assert_called_with(expected)
    mat_hi.assert_called_with(expected)


def test_predictor_dialog_no_selection_calls_set_highlight_none_both(qapp):
    """Clearing selection calls set_highlight(None) on both canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    with (
        patch.object(dialog._freq_canvas, "set_highlight") as freq_hi,
        patch.object(dialog._mat_canvas, "set_highlight") as mat_hi,
    ):
        dialog._table.clearSelection()
        dialog._on_selection_changed()

    freq_hi.assert_called_with(None)
    mat_hi.assert_called_with(None)


# ---------------------------------------------------------------------------
# Bus events
# ---------------------------------------------------------------------------


def test_predictor_dialog_on_predictor_changed_bus_event_refreshes_curves(qapp):
    """Bus PredictorChangedPayload with a loaded predictor must call predict_freq_curve."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert ctrl.predict_freq_curve.call_count == 0

    ctrl.get_predictor_info.return_value = {
        "path": "/new.json",
        "flux_bias": 0.0,
        "flux_half": 0.5,
        "flux_period": 1.0,
        "EJ": 4.0,
        "EC": 1.0,
        "EL": 1.0,
    }
    dialog._on_predictor_changed(object())

    assert ctrl.predict_freq_curve.call_count >= 1
    assert ctrl.predict_matrix_element_curve.call_count >= 1


def test_predictor_dialog_on_predictor_changed_cleared_clears_both_canvases(qapp):
    """Bus event with no predictor must blank both canvases."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    ctrl.get_predictor_info.return_value = None
    dialog._on_predictor_changed(object())

    assert dialog._freq_canvas._marker_line is None
    assert dialog._mat_canvas._marker_line is None


def test_predictor_dialog_on_predictor_changed_cleared_resets_both_columns(qapp):
    """After a cleared bus event, all Freq and |M| cells show '—'."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    ctrl.get_predictor_info.return_value = None
    dialog._on_predictor_changed(object())

    for row in range(dialog._table.rowCount()):
        freq_item = dialog._table.item(row, _COL_FREQ)
        assert freq_item is not None and freq_item.text() == "—"
        mag_item = dialog._table.item(row, _COL_MAG)
        assert mag_item is not None and mag_item.text() == "—"


def test_predictor_dialog_predict_curve_error_shown_in_status(qapp):
    """If predict_freq_curve raises, the error is shown in the status label."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    ctrl.predict_freq_curve.side_effect = PredictorNotLoaded("oops")
    dialog = PredictorDialog(ctrl)

    assert "oops" in dialog._status_label.text()
