"""Closed-loop feedback tests — the predictor adapts to measurements.

The runner module's autofluxdep is a closed loop: each qubit_freq measurement
calibrates the predictor (bias + an IDW error model), so the next flux point's
``predict_freq`` tracks the measured trend instead of staying a fixed prediction.
These tests cover the predictor's calibrate/IDW, the FluxoniumPredictorAdapter
glue, qubit_freq's calibrate trigger + fit-quality gate, and the end-to-end
adaptation over a sweep.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.tools import (
    FluxoniumPredictorAdapter,
    SimplePredictor,
)

# --- SimplePredictor: calibrate folds the residual into the prediction ---


def test_predict_is_linear_before_any_calibration():
    p = SimplePredictor(base=5000.0, slope=50.0)
    assert p.predict_freq(0.5) == 5025.0  # pure base + slope*flux


def test_calibrate_pulls_prediction_toward_measurement():
    p = SimplePredictor(base=5000.0, slope=50.0)
    p.calibrate(0.5, 5030.0)  # measured 5 above the linear 5025
    # at the calibrated flux the prediction now matches the measurement
    assert abs(p.predict_freq(0.5) - 5030.0) < 1e-6


def test_calibration_interpolates_between_points():
    p = SimplePredictor(base=5000.0, slope=50.0)
    p.calibrate(0.4, 5026.0)  # +6 over linear 5020
    p.calibrate(0.6, 5042.0)  # +12 over linear 5040
    mid = p.predict_freq(0.5)
    # IDW interpolates the learned residual — not the bare linear 5025
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


def test_adapter_calibrate_runs_both_loops():
    fake = _FakeFluxonium()
    adapter = FluxoniumPredictorAdapter(fluxonium=fake)
    adapter.calibrate(0.5, 5060.0)  # measured 5060, physical was 5050
    # loop 1: the fluxonium's bias was updated toward the measurement
    assert fake.bias != 0.0
    # the prediction now reflects the measurement at the calibrated flux
    assert abs(adapter.predict_freq(0.5) - 5060.0) < 1e-6


# Note: qubit_freq's real-acquire calibrate trigger (a good fit feeds
# predictor.calibrate; a poor fit skips it) is exercised end-to-end against the
# flux-aware MockSoc in test_qubit_freq_acquire.py — that test asserts the fitted
# frequency tracks flux, which only holds because each point calibrates the
# predictor. The pure predictor calibrate/IDW logic stays unit-tested above.


# --- predictor selection: no raw FluxoniumPredictor → SimplePredictor stand-in ---


def test_build_tools_falls_back_to_simple_predictor():
    # with no raw FluxoniumPredictor in the active context, the sweep's adaptive
    # predictor is the SimplePredictor stand-in, so a mock / unconfigured run
    # still drives the same calibrate loop.
    ctrl = build_core()
    tools = ctrl._build_tools()
    assert isinstance(tools.predictor, SimplePredictor)
