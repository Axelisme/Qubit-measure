"""Smoke tests for PredictorDialog."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.gui.session.services.connection import (
    LoadPredictorRequest,
    PredictCurveResult,
    PredictFreqRequest,
    PredictorLoadError,
    PredictorNotLoaded,
)
from zcu_tools.gui.session.ui.predictor_dialog import PredictorDialog


def _make_ctrl(
    *,
    has_predictor: bool = False,
    flux_half: float = 0.5,
    flux_period: float = 1.0,
    flux_bias: float = 0.0,
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
        }
    else:
        ctrl.get_predictor_info.return_value = None

    # Provide a valid PredictCurveResult so render_curves never raises.
    n = 20
    values = np.linspace(-0.5, 0.5, n)
    ctrl.predict_freq_curve.return_value = PredictCurveResult(
        labels=("0→1", "0→2", "0→3", "0→4"),
        values=values,
        fluxs=values.copy(),
        freqs_mhz=np.ones((4, n), dtype=np.float64) * 1000.0,
    )
    ctrl.predict_freq.return_value = 1234.5
    return ctrl


def test_predictor_dialog_init_and_load_dispatches_request(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = {
        "path": "/fake/path.json",
        "flux_bias": 0.5,
        "flux_half": 0.5,
        "flux_period": 1.0,
    }

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


def test_predictor_dialog_predict_dispatches_request(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    ctrl.predict_freq.return_value = 123.456

    dialog = PredictorDialog(ctrl)
    dialog._predict_value_spin.setValue(1.5)
    dialog._from_spin.setValue(1)
    dialog._to_spin.setValue(2)
    dialog._on_predict_clicked()

    ctrl.predict_freq.assert_called_once()
    (req,) = ctrl.predict_freq.call_args.args
    assert isinstance(req, PredictFreqRequest)
    assert req.value == 1.5
    assert req.transition == (1, 2)
    assert dialog._predict_result_label.text() == "123.4560 MHz"


def test_predictor_dialog_predict_without_predictor_shows_status(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    ctrl.predict_freq.side_effect = PredictorNotLoaded("nothing loaded")

    dialog = PredictorDialog(ctrl)
    dialog._on_predict_clicked()

    assert "nothing loaded" in dialog._status_label.text()


def test_predictor_dialog_clear(qapp):
    ctrl = MagicMock()
    ctrl.get_predictor_info.return_value = None
    dialog = PredictorDialog(ctrl)

    dialog._on_clear()
    ctrl.clear_predictor.assert_called_once()
    assert dialog._predict_result_label.text() == "—"


# ---------------------------------------------------------------------------
# New tests for the canvas integration
# ---------------------------------------------------------------------------


def test_predictor_dialog_init_with_predictor_calls_predict_curve(qapp):
    """When a predictor is already loaded at init, predict_freq_curve is called."""
    ctrl = _make_ctrl(
        has_predictor=True, path="/p.json", flux_half=0.5, flux_period=1.0
    )
    dialog = PredictorDialog(ctrl)
    # predict_freq_curve must have been called once during _refresh_curves.
    ctrl.predict_freq_curve.assert_called_once()


def test_predictor_dialog_init_no_predictor_no_curve_call(qapp):
    """With no predictor at init, predict_freq_curve is never called."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    ctrl.predict_freq_curve.assert_not_called()


def test_predictor_dialog_init_with_predictor_canvas_has_axes(qapp):
    """After init with a loaded predictor, the canvas has axes (curve was drawn)."""
    ctrl = _make_ctrl(
        has_predictor=True, path="/p.json", flux_half=0.5, flux_period=1.0
    )
    dialog = PredictorDialog(ctrl)
    assert len(dialog._canvas.figure.get_axes()) > 0


def test_predictor_dialog_spinbox_change_does_not_recompute_curves(qapp):
    """Changing the Flux value spinbox must NOT call predict_freq_curve again."""
    ctrl = _make_ctrl(
        has_predictor=True, path="/p.json", flux_half=0.5, flux_period=1.0
    )
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    # Simulate spinbox change via the internal slot (skip Qt event loop).
    dialog._on_spinbox_changed(0.3)

    # No extra curve computation triggered.
    assert ctrl.predict_freq_curve.call_count == initial_count


def test_predictor_dialog_canvas_drag_updates_spinbox_no_loop(qapp):
    """Canvas on_drag must update the spinbox without triggering infinite recursion."""
    ctrl = _make_ctrl(
        has_predictor=True, path="/p.json", flux_half=0.5, flux_period=1.0
    )
    dialog = PredictorDialog(ctrl)

    # Track setValue calls by checking the value afterwards.
    dialog._on_canvas_drag(0.77)
    assert dialog._predict_value_spin.value() == pytest.approx(0.77)


def test_predictor_dialog_transition_change_does_not_recompute_curves(qapp):
    """Changing transition spinboxes must NOT call predict_freq_curve again.

    Only set_highlight is expected — the curve data is unchanged; only the
    highlight style on the existing artists needs updating.
    """
    from unittest.mock import patch

    ctrl = _make_ctrl(
        has_predictor=True, path="/p.json", flux_half=0.5, flux_period=1.0
    )
    dialog = PredictorDialog(ctrl)
    initial_count = ctrl.predict_freq_curve.call_count

    with patch.object(dialog._canvas, "set_highlight") as mock_set_highlight:
        dialog._from_spin.setValue(0)
        dialog._to_spin.setValue(2)
        dialog._on_transition_changed()

    # No extra curve computation.
    assert ctrl.predict_freq_curve.call_count == initial_count
    # set_highlight was called (spinboxes fire valueChanged each; at least 1 call expected).
    assert mock_set_highlight.call_count >= 1
    # The final explicit call must carry the current (from, to) pair.
    mock_set_highlight.assert_called_with((0, 2))


def test_predictor_dialog_clear_blanks_canvas(qapp):
    """After _on_clear, the canvas must have no marker line."""
    ctrl = _make_ctrl(
        has_predictor=True, path="/p.json", flux_half=0.5, flux_period=1.0
    )
    dialog = PredictorDialog(ctrl)
    dialog._on_clear()

    # After clear, canvas internal state is reset.
    assert dialog._canvas._marker_value is None


def test_predictor_dialog_on_predictor_changed_bus_event_refreshes_curves(qapp):
    """Bus PredictorChangedPayload with a loaded predictor must call predict_freq_curve."""
    ctrl = _make_ctrl(has_predictor=False)
    dialog = PredictorDialog(ctrl)
    assert ctrl.predict_freq_curve.call_count == 0

    # Simulate the predictor being loaded (bus event).
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
    ctrl = _make_ctrl(
        has_predictor=True, path="/p.json", flux_half=0.5, flux_period=1.0
    )
    dialog = PredictorDialog(ctrl)

    # Simulate predictor cleared.
    ctrl.get_predictor_info.return_value = None
    dialog._on_predictor_changed(object())

    assert dialog._canvas._marker_line is None


def test_predictor_dialog_predict_curve_error_shown_in_status(qapp):
    """If predict_freq_curve raises, the error is shown in the status label."""
    ctrl = _make_ctrl(
        has_predictor=True, path="/p.json", flux_half=0.5, flux_period=1.0
    )
    ctrl.predict_freq_curve.side_effect = PredictorNotLoaded("oops")
    dialog = PredictorDialog(ctrl)

    # _refresh_curves is called during init; check that the status shows the error.
    assert "oops" in dialog._status_label.text()
