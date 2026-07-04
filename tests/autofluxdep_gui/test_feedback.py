"""Closed-loop feedback tests — prediction correction is explicit feedback.

The runner module's autofluxdep is a closed loop: each qubit_freq measurement
calibrates the physical predictor when one exists and feeds a generic residual
estimator, so the next flux point's node-owned composed prediction tracks the
measured trend instead of hiding IDW inside the predictor.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.feedback import IdwEstimator
from zcu_tools.gui.app.autofluxdep.tools import (
    FluxoniumPredictorAdapter,
    SimplePredictor,
)

# --- SimplePredictor: base fallback, no hidden residual correction ---


def test_predict_is_linear_before_any_calibration():
    p = SimplePredictor(base=5000.0, slope=50.0)
    assert p.predict_freq(0.5) == 5025.0  # pure base + slope*flux


def test_simple_predictor_calibrate_keeps_base_prediction_unchanged():
    p = SimplePredictor(base=5000.0, slope=50.0)
    p.calibrate(0.5, 5030.0)  # measured 5 above the linear 5025
    assert p.predict_freq(0.5) == 5025.0


def test_idw_feedback_estimator_interpolates_prediction_residuals():
    base = SimplePredictor(base=5000.0, slope=50.0)
    estimator = IdwEstimator()
    estimator.observe(0.4, 5026.0 - base.predict_freq(0.4))  # +6 residual
    estimator.observe(0.6, 5042.0 - base.predict_freq(0.6))  # +12 residual
    correction = estimator.estimate(0.5)
    assert correction is not None
    mid = base.predict_freq(0.5) + correction
    assert mid != 5025.0
    assert 5025.0 < mid < 5040.0


# --- FluxoniumPredictorAdapter: same interface over a fake fluxonium ---


class _FakeFluxonium:
    """A minimal stand-in exposing the FluxoniumPredictor surface."""

    def __init__(self):
        self.bias = 0.0

    def predict_freq(self, flux):
        return 5000.0 + 100.0 * flux + self.bias

    def predict_matrix_element(self, flux):
        del flux
        return 0.2

    def calculate_bias(self, flux, measured):
        return measured - (5000.0 + 100.0 * flux)  # the bias that hits measured

    def update_bias(self, bias):
        self.bias = bias


def test_adapter_exposes_predictor_interface():
    adapter = FluxoniumPredictorAdapter(fluxonium=_FakeFluxonium())
    assert adapter.predict_freq(0.3) == 5030.0  # 5000 + 30 + 0 bias + 0 idw
    assert adapter.predict_matrix_element(0.3) == 0.2


def test_adapter_calibrate_runs_physical_loop():
    fake = _FakeFluxonium()
    adapter = FluxoniumPredictorAdapter(fluxonium=fake)
    adapter.calibrate(0.5, 5060.0)  # measured 5060, physical was 5050
    assert fake.bias != 0.0
    # the prediction now reflects the measurement at the calibrated flux
    assert abs(adapter.predict_freq(0.5) - 5060.0) < 1e-6


# Note: qubit_freq's real-acquire feedback trigger (a good fit feeds predictor
# calibration and residual estimator; a poor fit skips both) is exercised
# end-to-end against the flux-aware MockSoc in test_qubit_freq_acquire.py.


# --- predictor selection: no raw FluxoniumPredictor → SimplePredictor stand-in ---


def test_build_tools_falls_back_to_simple_predictor():
    # with no raw FluxoniumPredictor in the active context, the sweep's base
    # predictor is the SimplePredictor stand-in.
    ctrl = build_core()
    tools = ctrl._build_tools()
    assert isinstance(tools.predictor, SimplePredictor)
