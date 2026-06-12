"""RB-2: t1 end-to-end real acquire against the flux-aware MockSoc.

Runs the t1 Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(Reset, pi_pulse, Delay, Readout) over the relax-time sweep ->
fit_decay) and asserts a finite, positive T1 comes back -- i.e. the real acquire
produced a fittable exponential decay (not noise / not a constant).

The pi_pulse excites the qubit at the predicted f01 (matching DEFAULT_SIMPARAM) so
the relax sweep traces out a real decay whose constant fit_decay recovers.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from ._helpers import ACQUIRE_READOUT, connect_mock, make_acquire_env

# reps=1000 averages the mock per-shot noise enough that the decay clears the
# fit-quality gate (200 reps leaves residual > the gate at this T1 contrast).
_PARAMS = {"sweep_range": "0.5,60,41", "reps": 1000, "rounds": 1}


def _pi_pulse(ml, freq: float):
    ml.register_waveform(t1_pi={"style": "const", "length": 1.0})
    return {
        "type": "pulse",
        "waveform": ml.get_waveform("t1_pi", {"length": 0.1}),
        "ch": 1,
        "nqz": 1,
        "gain": 0.5,
        "freq": freq,
    }


def test_t1_acquire_fits_finite_positive_t1():
    ctrl = build_core()
    connect_mock(ctrl)
    ml = ctrl.state.exp_context.ml
    predictor = FluxoniumPredictor(
        params=(4.0, 1.0, 1.0), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )

    builder = T1Builder()
    flux_values = [0.0, 0.1]
    t1s: list[float] = []
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(_PARAMS, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, params=_PARAMS, ml=ml, result=result
        )
        f01 = float(predictor.predict_freq(flux))
        snap = Snapshot(
            {"t1": 10.0},
            modules={"pi_pulse": _pi_pulse(ml, f01), "opt_readout": ACQUIRE_READOUT},
        )
        patch = builder.build_node(env).produce(snap)
        if "t1" in patch.values():
            t1s.append(float(patch.values()["t1"]))

    assert t1s, "no flux point fit a T1 from the real acquire"
    assert all(np.isfinite(t) and t > 0.0 for t in t1s), f"non-physical T1: {t1s}"
