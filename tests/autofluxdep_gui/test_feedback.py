"""Closed-loop feedback tests — the predictor adapts to measurements.

The runner module's autofluxdep is a closed loop: each qubit_freq measurement
calibrates the predictor (bias + an IDW error model), so the next flux point's
``predict_freq`` tracks the measured trend instead of staying a fixed prediction.
These tests cover the predictor's calibrate/IDW, the FluxoniumPredictorAdapter
glue, qubit_freq's calibrate trigger + fit-quality gate, and the end-to-end
adaptation over a sweep.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.tools import (
    FluxoniumPredictorAdapter,
    SimplePredictor,
    Tools,
)

from ._helpers import connect_mock

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


# --- qubit_freq triggers calibrate on a good fit, skips it on a poor one ---


def _produce_once(predictor, flux=0.0, predict_freq=5000.0):
    builder = QubitFreqBuilder()
    result = builder.make_init_result({"detune_sweep": "-20,50,0.5"}, np.array([flux]))
    env = RunEnv(
        flux=flux,
        flux_idx=0,
        params={"rounds": 2, "acquire_delay": 0},
        tools=Tools(predictor=predictor),
        result=result,
    )
    node = builder.build_node(env)
    snap = Snapshot(
        {"predict_freq": predict_freq, "fit_kappa": 0.05}, modules={"readout": None}
    )
    return node.produce(snap), result


def test_good_fit_calibrates_the_predictor():
    p = SimplePredictor(base=5000.0, slope=50.0)
    before = p.predict_freq(0.0)
    patch, _result = _produce_once(p, flux=0.0, predict_freq=5000.0)
    assert "qubit_freq" in patch.values()  # produced
    # the predictor was calibrated → its prediction at flux 0 moved toward measured
    assert p.predict_freq(0.0) != before


def test_calibrate_uses_env_flux_and_measured_freq():
    # the calibration uses env.flux (not the snapshot) and the fitted freq
    p = SimplePredictor(base=5000.0, slope=50.0)
    patch, _result = _produce_once(p, flux=0.7, predict_freq=5035.0)
    measured = patch.values()["qubit_freq"]
    # after calibrate(0.7, measured), the prediction at 0.7 equals the measurement
    assert abs(p.predict_freq(0.7) - measured) < 1e-6


def test_produce_without_tools_does_not_crash():
    # a headless produce with no predictor (env.tools None) must still fit + emit
    builder = QubitFreqBuilder()
    result = builder.make_init_result({"detune_sweep": "-20,50,0.5"}, np.array([0.0]))
    env = RunEnv(flux=0.0, flux_idx=0, params={"rounds": 2}, result=result)
    patch = builder.build_node(env).produce(
        Snapshot({"predict_freq": 5000.0, "fit_kappa": 0.05}, modules={"readout": None})
    )
    assert "qubit_freq" in patch.values()  # fit still produced, just no feedback


# --- end-to-end: predict_freq adapts across a sweep ---


def test_sweep_adapts_prediction_to_measurements():
    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    for node in ctrl.state.nodes:
        node.params["acquire_delay"] = 0
        node.params["rounds"] = 2
    connect_mock(ctrl)
    ctrl.set_flux_values([0.0, 0.2, 0.4, 0.6, 0.8])
    ctrl.start_run()

    predictor = ctrl.state.run_predictor
    assert predictor is not None
    # the synthetic resonance sits 1.5 MHz above the physical base; after the
    # sweep calibrated every point, the prediction tracks the measured (~base+1.5)
    # rather than the bare linear base. Check a calibrated flux moved.
    physical_base_at_04 = 5000.0 + 50.0 * 0.4  # 5020
    assert abs(predictor.predict_freq(0.4) - physical_base_at_04) > 0.8


# --- SNR-trough dead points are skipped (no key, no calibrate) ---


def test_sweep_skips_snr_trough_dead_points():
    # the SNR varies to 0 at troughs; those flux points are pure noise, so the
    # fit-quality gate discards them — fit_freq stays nan and the predictor is
    # not calibrated there, but the sweep keeps going (good points still tracked).
    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    for node in ctrl.state.nodes:
        node.params["acquire_delay"] = 0
        node.params["rounds"] = 4
    connect_mock(ctrl)
    ctrl.set_flux_values(list(np.linspace(0.0, 1.0, 11)))
    ctrl.start_run()

    res = ctrl.state.run_results["qubit_freq"]
    n_dead = int(np.sum(np.isnan(res.fit_freq)))
    n_good = int(np.sum(~np.isnan(res.fit_freq)))
    assert n_dead >= 1  # at least one SNR-trough dead point was rejected
    assert n_good >= 5  # most points still fit + drove feedback
