"""Tests for PredictorDialog — tracked-transitions table UX."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from zcu_tools.gui.session.services.connection import (
    LoadPredictorRequest,
    PredictCurveResult,
    PredictorLoadError,
    PredictorNotLoaded,
)
from zcu_tools.gui.session.ui.predictor_dialog import (
    _DEFAULT_TRANSITIONS,
    PredictorDialog,
)


def _make_result(n_transitions: int, n: int = 20) -> PredictCurveResult:
    """Build a minimal PredictCurveResult for ``n_transitions`` curves."""
    transitions = _DEFAULT_TRANSITIONS[:n_transitions]
    values = np.linspace(-0.5, 0.5, n)
    labels = tuple(f"{f}→{t}" for f, t in transitions)
    freqs = np.ones((n_transitions, n), dtype=np.float64) * 1000.0
    return PredictCurveResult(
        labels=labels,
        values=values,
        fluxs=values.copy(),
        freqs_mhz=freqs,
    )


def _make_ctrl(
    *,
    has_predictor: bool = False,
    flux_half: float = 0.5,
    flux_period: float = 1.0,
    flux_bias: float = 0.0,
    path: str | None = None,
    n_transitions: int = 5,
) -> MagicMock:
    """Build a MagicMock controller pre-configured for dialog tests."""
    ctrl = MagicMock()
    if has_predictor:
        ctrl.get_predictor_info.return_value = {
            "path": path,
            "flux_bias": flux_bias,
            "flux_half": flux_half,
            "flux_period": flux_period,
        }
    else:
        ctrl.get_predictor_info.return_value = None

    # Return a result whose transition count matches whatever is tracked.
    # The mock simply returns a fresh result on every call.
    def _make_curve_result(req):  # type: ignore[no-untyped-def]
        n = len(req.transitions)
        values = np.linspace(-0.5, 0.5, 20)
        labels = tuple(f"{f}→{t}" for f, t in req.transitions)
        freqs = np.ones((n, 20), dtype=np.float64) * 1000.0
        return PredictCurveResult(
            labels=labels,
            values=values,
            fluxs=values.copy(),
            freqs_mhz=freqs,
        )

    ctrl.predict_freq_curve.side_effect = _make_curve_result
    ctrl.predict_freq.return_value = 1234.5
    return ctrl


# ---------------------------------------------------------------------------
# Init / load / clear
# ---------------------------------------------------------------------------


def test_predictor_dialog_init_and_load_dispatches_request(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = {
        "path": "/fake/path.json",
        "flux_bias": 0.5,
        "flux_half": 0.5,
        "flux_period": 1.0,
    }
    # predict_freq must return a float so _update_freq_column can format it.
    ctrl.predict_freq.return_value = 1234.5

    # predict_freq_curve returns a minimal result matching the tracked set.
    def _make_curve_result(req):  # type: ignore[no-untyped-def]
        n = len(req.transitions)
        values = np.linspace(-0.5, 0.5, 20)
        labels = tuple(f"{f}→{t}" for f, t in req.transitions)
        freqs = np.ones((n, 20), dtype=np.float64) * 1000.0
        return PredictCurveResult(
            labels=labels,
            values=values,
            fluxs=values.copy(),
            freqs_mhz=freqs,
        )

    ctrl.predict_freq_curve.side_effect = _make_curve_result

    dialog = PredictorDialog(ctrl)

    assert dialog._path_edit.text() == "/fake/path.json"
    assert dialog._flux_bias_spin.value() == 0.5

    dialog._path_edit.setText("new.json")
    dialog._on_accepted()

    ctrl.load_predictor.assert_called_once()
    (req,) = ctrl.load_predictor.call_args.args
    assert isinstance(req, LoadPredictorRequest)
    assert req.path == "new.json"
    assert req.flux_bias == 0.5


def test_predictor_dialog_load_failure_shown_in_status(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    ctrl.load_predictor.side_effect = PredictorLoadError("bad file")

    dialog = PredictorDialog(ctrl)
    dialog._path_edit.setText("missing.json")
    dialog._on_accepted()

    assert "bad file" in dialog._status_label.text()


def test_predictor_dialog_clear(qapp):
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    dialog._on_clear()
    ctrl.clear_predictor.assert_called_once()
    # All Freq cells must show "—" after clear.
    for row in range(dialog._table.rowCount()):
        item = dialog._table.item(row, 1)
        assert item is not None and item.text() == "—"


# ---------------------------------------------------------------------------
# Default tracked transitions — table layout
# ---------------------------------------------------------------------------


def test_predictor_dialog_default_tracked_count(qapp):
    """Dialog initialises with 5 default tracked transitions, table has 5 rows."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert len(dialog._tracked) == 5
    assert dialog._table.rowCount() == 5


def test_predictor_dialog_default_predict_freq_curve_call(qapp):
    """With predictor loaded at init, predict_freq_curve is called with the 5 defaults."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    PredictorDialog(ctrl)
    ctrl.predict_freq_curve.assert_called()
    last_req = ctrl.predict_freq_curve.call_args.args[0]
    assert set(last_req.transitions) == set(map(tuple, _DEFAULT_TRANSITIONS))


def test_predictor_dialog_init_no_predictor_no_curve_call(qapp):
    """With no predictor at init, predict_freq_curve is never called."""
    ctrl = _make_ctrl(has_predictor=False)
    PredictorDialog(ctrl)
    ctrl.predict_freq_curve.assert_not_called()


def test_predictor_dialog_init_with_predictor_canvas_has_axes(qapp):
    """After init with a loaded predictor, the canvas has axes (curve was drawn)."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    assert len(dialog._canvas.figure.get_axes()) > 0


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
    # Curve refresh triggered.
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
# Delete transition
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


def test_predictor_dialog_delete_all_transitions_clears_canvas(qapp):
    """Deleting all tracked transitions results in an empty table and cleared canvas."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    for t in list(dialog._tracked):
        dialog._delete_transition(t)

    assert dialog._table.rowCount() == 0
    # Canvas clear() resets marker_value.
    assert dialog._canvas._marker_value is None


# ---------------------------------------------------------------------------
# Spinbox / debounce → freq column update
# ---------------------------------------------------------------------------


def test_predictor_dialog_spinbox_change_does_not_recompute_curves(qapp):
    """Changing the Flux value spinbox must NOT call predict_freq_curve again."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    dialog._on_spinbox_changed(0.3)

    assert ctrl.predict_freq_curve.call_count == initial_count


def test_predictor_dialog_freq_column_updated_after_debounce(qapp):
    """After _update_freq_column, each tracked row's Freq cell is filled."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    dialog._predict_value_spin.setValue(0.1)
    dialog._update_freq_column()

    # Each row should now show a numeric freq (not "—").
    for row in range(dialog._table.rowCount()):
        item = dialog._table.item(row, 1)
        assert item is not None
        assert item.text() != "—"
    # predict_freq called at least once per tracked transition in this call.
    # (init also calls _update_freq_column once, so total >= 2×n.)
    assert ctrl.predict_freq.call_count >= len(dialog._tracked)


def test_predictor_dialog_freq_column_dash_when_not_loaded(qapp):
    """predict_freq raising PredictorNotLoaded → Freq cell shows '—', no crash."""
    ctrl = _make_ctrl(has_predictor=False)
    ctrl.predict_freq.side_effect = PredictorNotLoaded("no predictor")
    dialog = PredictorDialog(ctrl)

    dialog._update_freq_column()

    for row in range(dialog._table.rowCount()):
        item = dialog._table.item(row, 1)
        assert item is not None
        assert item.text() == "—"


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
# Table selection → canvas highlight
# ---------------------------------------------------------------------------


def test_predictor_dialog_row_select_calls_set_highlight(qapp):
    """Selecting a table row calls canvas.set_highlight with the right transition."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    with patch.object(dialog._canvas, "set_highlight") as mock_hi:
        dialog._table.selectRow(1)
        dialog._on_selection_changed()

    expected = dialog._tracked[1]
    mock_hi.assert_called_with(expected)


def test_predictor_dialog_no_selection_calls_set_highlight_none(qapp):
    """Clearing selection calls canvas.set_highlight(None)."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    with patch.object(dialog._canvas, "set_highlight") as mock_hi:
        dialog._table.clearSelection()
        dialog._on_selection_changed()

    mock_hi.assert_called_with(None)


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
    }
    dialog._on_predictor_changed(object())

    assert ctrl.predict_freq_curve.call_count >= 1


def test_predictor_dialog_on_predictor_changed_cleared_clears_canvas(qapp):
    """Bus event with no predictor must blank the canvas."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    ctrl.get_predictor_info.return_value = None
    dialog._on_predictor_changed(object())

    assert dialog._canvas._marker_line is None


def test_predictor_dialog_on_predictor_changed_cleared_resets_freq_column(qapp):
    """After a cleared bus event, all Freq cells show '—'."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    dialog = PredictorDialog(ctrl)

    ctrl.get_predictor_info.return_value = None
    dialog._on_predictor_changed(object())

    for row in range(dialog._table.rowCount()):
        item = dialog._table.item(row, 1)
        assert item is not None and item.text() == "—"


def test_predictor_dialog_predict_curve_error_shown_in_status(qapp):
    """If predict_freq_curve raises, the error is shown in the status label."""
    ctrl = _make_ctrl(has_predictor=True, path="/p.json")
    ctrl.predict_freq_curve.side_effect = PredictorNotLoaded("oops")
    dialog = PredictorDialog(ctrl)

    assert "oops" in dialog._status_label.text()
