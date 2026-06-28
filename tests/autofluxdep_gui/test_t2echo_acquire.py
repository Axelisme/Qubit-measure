"""RB-2: t2echo end-to-end real acquire against the flux-aware MockSoc.

Runs the t2echo Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(Reset, pi/2, Delay, pi, Delay, detuned pi/2, Readout) over the
total-delay sweep -> fit_decay_fringe) and asserts a finite, positive T2Echo comes
back -- i.e. the real Hahn-echo acquire produced a fittable decaying fringe.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import SweepValue
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from ._helpers import (
    ACQUIRE_READOUT,
    connect_mock,
    make_acquire_env,
    node_schema,
)

# reps=2000 x rounds=2 averages the mock per-shot noise enough that the echo
# fringe clears the fit-quality gate (the echo contrast is lower than T1's).
# sweep_range overrides the schema default's 121 pts → 61 pts for test speed;
# the production default stays at 121 in the node schema.
_PARAMS = {
    "reps": 2000,
    "rounds": 2,
    "detune_ratio": 0.2,
    "sweep_range": SweepValue(start=0.0, stop=25.0, expts=61),
}


def _pulses(ml, freq: float):
    ml.register_waveform(echo_drive={"style": "const", "length": 1.0})
    pi_pulse = {
        "type": "pulse",
        "waveform": ml.get_waveform("echo_drive", {"length": 0.1}),
        "ch": 1,
        "nqz": 1,
        "gain": 0.5,
        "freq": freq,
    }
    pi2_pulse = {
        "type": "pulse",
        "waveform": ml.get_waveform("echo_drive", {"length": 0.05}),
        "ch": 1,
        "nqz": 1,
        "gain": 0.25,
        "freq": freq,
    }
    return pi_pulse, pi2_pulse


def test_t2echo_acquire_fits_finite_positive_t2():
    ctrl = build_core()
    connect_mock(ctrl)
    ml = ctrl.state.exp_context.ml
    predictor = FluxoniumPredictor(
        params=(4.0, 1.0, 1.0), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )

    builder = T2EchoBuilder()
    schema = node_schema(builder, _PARAMS)
    flux_values = [0.0, 0.1]
    t2s: list[float] = []
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(schema, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, schema=schema, ml=ml, result=result
        )
        f01 = float(predictor.predict_freq(flux))
        pi_pulse, pi2_pulse = _pulses(ml, f01)
        snap = Snapshot(
            {"t1": 20.0, "t2e": 8.0},
            modules={
                "pi_pulse": pi_pulse,
                "pi2_pulse": pi2_pulse,
                "opt_readout": ACQUIRE_READOUT,
            },
        )
        patch = builder.build_node(env).produce(snap)
        if "t2e" in patch.values():
            t2s.append(float(patch.values()["t2e"]))

    assert t2s, "no flux point fit a T2Echo from the real acquire"
    assert all(np.isfinite(t) and t > 0.0 for t in t2s), f"non-physical T2e: {t2s}"
