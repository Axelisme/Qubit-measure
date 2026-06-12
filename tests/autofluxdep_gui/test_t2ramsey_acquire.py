"""RB-2: t2ramsey end-to-end real acquire against the flux-aware MockSoc.

Runs the t2ramsey Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(Reset, pi/2, Delay, detuned pi/2, Readout) over the delay sweep ->
fit_decay_fringe) and asserts a finite, positive T2Ramsey comes back -- i.e. the
real acquire produced a fittable decaying fringe.

The pi/2 pulses drive at the predicted f01; an activate-detune phase ramp on the
second pulse makes the fringe resolvable (lower-layer behaviour).
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from ._helpers import ACQUIRE_READOUT, connect_mock, make_acquire_env

# sweep_range overrides _DEFAULT_SWEEP's 121 pts → 61 pts for test speed;
# the production default stays at 121 in the node module.
_PARAMS = {"reps": 1000, "rounds": 1, "detune_ratio": 0.2, "sweep_range": "0.0,25.0,61"}


def _pi2_pulse(ml, freq: float):
    ml.register_waveform(ramsey_pi2={"style": "const", "length": 1.0})
    return {
        "type": "pulse",
        "waveform": ml.get_waveform("ramsey_pi2", {"length": 0.05}),
        "ch": 1,
        "nqz": 1,
        "gain": 0.25,
        "freq": freq,
    }


def test_t2ramsey_acquire_fits_finite_positive_t2():
    ctrl = build_core()
    connect_mock(ctrl)
    ml = ctrl.state.exp_context.ml
    predictor = FluxoniumPredictor(
        params=(4.0, 1.0, 1.0), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )

    builder = T2RamseyBuilder()
    flux_values = [0.0, 0.1]
    t2s: list[float] = []
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(_PARAMS, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, params=_PARAMS, ml=ml, result=result
        )
        f01 = float(predictor.predict_freq(flux))
        snap = Snapshot(
            {"t1": 10.0, "t2r": 8.0},
            modules={"pi2_pulse": _pi2_pulse(ml, f01), "opt_readout": ACQUIRE_READOUT},
        )
        patch = builder.build_node(env).produce(snap)
        if "t2r" in patch.values():
            t2s.append(float(patch.values()["t2r"]))

    assert t2s, "no flux point fit a T2Ramsey from the real acquire"
    assert all(np.isfinite(t) and t > 0.0 for t in t2s), f"non-physical T2r: {t2s}"
