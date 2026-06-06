"""qubit_freq Builder tests — synthesise → real fit → fill Result → Patch.

The qubit_freq Node synthesises a Lorentzian dip and fits it with the real
``fit_qubit_freq``; the fitted freq/kappa must recover the planted values, the
sweep Result's row must be filled in place, and the round_hook must fire. This
exercises the full build_node → produce → fit → fill → notify path without
hardware (MockSoc gives only noise, so the Node synthesises).
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder


def _produce(predict_freq: float, flux=0.0, flux_idx: int = 0, sweep="-20,50,0.5"):
    """Build the Result + Node for one flux point and run produce.

    ``flux`` is the device flux value (in [0,1]); the true resonance drifts with
    it (``flux_drift``) and the SNR varies (``flux_snr``), so pick a flux away
    from an SNR trough for the recovery tests. ``flux_idx`` is the Result row.
    """
    builder = QubitFreqBuilder()
    result = builder.make_init_result({"detune_sweep": sweep}, np.linspace(0.0, 1.0, 8))
    fired: list = []
    env = RunEnv(
        flux=flux,
        flux_idx=flux_idx,
        params={"detune_sweep": sweep},
        result=result,
        round_hook=lambda payload: fired.append(payload),
    )
    node = builder.build_node(env)
    snap = Snapshot(
        {"predict_freq": predict_freq, "fit_kappa": 0.05},
        modules={"readout": "<ro>"},
    )
    patch = node.produce(snap)
    return patch, result, fired


def _planted_freq(predict_freq: float, flux: float) -> float:
    """The drifted true resonance qubit_freq's produce plants at ``flux``."""
    from zcu_tools.gui.app.autofluxdep.nodes.synth import flux_drift

    return predict_freq + flux_drift(flux, baseline=1.5, amplitude=20.0)


def test_produce_recovers_planted_freq_and_kappa():
    # at the sweet-spot flux 0.5 the drift offset is just the baseline 1.5
    patch, _result, _fired = _produce(5000.0, flux=0.5)
    vals = patch.values()
    expected = _planted_freq(5000.0, 0.5)  # 5000 + 1.5
    assert abs(vals["qubit_freq"] - expected) < 0.5
    assert abs(vals["fit_detune"] - 1.5) < 0.5
    assert abs(vals["fit_kappa"] - 2.0) < 0.6


def test_produce_emits_exactly_the_declared_keys():
    patch, _result, _fired = _produce(5000.0, flux=0.5)
    assert set(patch.values()) == {"qubit_freq", "fit_detune", "fit_kappa"}
    assert patch.modules() == {}  # qubit_freq produces no module


def test_produce_fills_the_result_row_in_place():
    patch, result, _fired = _produce(5000.0, flux=0.5, flux_idx=3)
    assert "qubit_freq" in patch.values()  # a good fit at the sweet spot
    # the produced row carries flux, predict_freq, signal, fit curve + fit_freq
    assert result.flux[3] == 0.5
    assert result.predict_freq[3] == 5000.0
    assert not np.isnan(result.signal[3]).any()
    assert not np.isnan(result.fit_curve[3]).any()
    assert abs(result.fit_freq[3] - _planted_freq(5000.0, 0.5)) < 0.5
    # untouched rows stay nan (honest "not measured")
    assert np.isnan(result.fit_freq[0])
    assert np.isnan(result.signal[0]).all()


def test_produce_fires_round_hook_for_liveplot():
    _patch, _result, fired = _produce(5000.0)
    # the Node notifies at least once (raw filled) — drives the main-thread redraw
    assert len(fired) >= 1


def test_produce_tracks_the_predicted_frequency():
    # the resonance sits 1.5 MHz above the prediction → fitted freq follows it
    a = _produce(5000.0)[0].values()["qubit_freq"]
    b = _produce(5200.0)[0].values()["qubit_freq"]
    assert abs((b - a) - 200.0) < 1.0


def test_produce_deterministic_per_flux_index():
    # same flux index → same synthetic noise seed → identical fit
    assert _produce(5000.0, flux_idx=3)[0].values() == (
        _produce(5000.0, flux_idx=3)[0].values()
    )


def test_controller_run_drives_real_produce_with_predictor_service():
    # a full controller run (no Qt) drives the predictor Service + qubit_freq's
    # produce and leaves a fitted qubit_freq in the final InfoStore.
    from zcu_tools.gui.app.autofluxdep.app import build_core

    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    ctrl.state.nodes[0].params["acquire_delay"] = 0  # instant headless run
    ctrl.setup(use_mock=True)
    ctrl.set_flux_values([0.0, 1.0])
    info = ctrl.start_run()
    # the predictor Service produced predict_freq + closed-loop feedback adapted
    # it: the drifting resonance is measured and calibrated, so flux 1.0's predict
    # is above the bare linear 5050, and qubit_freq sits in the detune window above.
    assert info.point["predict_freq"] > 5050.0  # adapted by feedback
    offset = info.point["qubit_freq"] - info.point["predict_freq"]
    assert 0.0 < offset < 50.0  # the drifted resonance, within the sweep window
    assert info.point["qubit_freq"] != 1.0  # not the fake fallback
