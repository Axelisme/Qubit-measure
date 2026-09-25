"""RB-2: t2echo end-to-end real acquire against the flux-aware MockSoc.

Runs the t2echo Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(Reset, pi/2, Delay, pi, Delay, pi/2, Readout) over the
total-delay sweep -> fit_decay) and asserts a finite, positive T2Echo comes back
from a fit that clears the legacy feedback-success gate.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.experiments.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.cfg import SweepValue

from .._helpers import (
    ACQUIRE_READOUT,
    calibrated_drive_pulse,
    connect_mock,
    high_snr_simparams,
    make_acquire_env,
    mock_flux_predictor,
    node_schema,
)

# sweep_range overrides the schema default's 121 pts → 61 pts for test speed;
# the production default stays at 121 in the node schema.
_PARAMS = {
    "reps": 2000,
    "rounds": 2,
    "earlystop_snr": None,
    "sweep_range_mode": "fixed",
    "relax_delay_mode": "fixed",
    "fit_method": "decay",
    "relax_delay": 60.0,
    "detune_ratio": 0.0,
    "sweep_range": SweepValue(start=0.0, stop=25.0, expts=61),
}


def _pulses(ml, freq: float, sim_params):
    pi_pulse = calibrated_drive_pulse(ml, "echo_pi", freq, sim_params=sim_params)
    pi2_pulse = calibrated_drive_pulse(
        ml, "echo_pi2", freq, sim_params=sim_params, angle=0.5
    )
    return pi_pulse, pi2_pulse


def test_t2echo_acquire_fits_finite_positive_t2():
    ctrl = build_core()
    sim_params = high_snr_simparams()
    connect_mock(ctrl, sim_params=sim_params)
    ml = ctrl.state.exp_context.ml
    predictor = mock_flux_predictor(sim_params)

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
        pi_pulse, pi2_pulse = _pulses(ml, f01, sim_params)
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
