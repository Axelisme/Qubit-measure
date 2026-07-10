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
from zcu_tools.gui.cfg import SweepValue

from ._helpers import (
    ACQUIRE_READOUT,
    calibrated_drive_pulse,
    connect_mock,
    high_snr_simparams,
    make_acquire_env,
    mock_flux_predictor,
    node_schema,
)

_PARAMS = {
    "sweep_range": SweepValue(start=0.5, stop=60.0, expts=41),
    "sweep_range_mode": "fixed",
    "relax_delay_mode": "fixed",
    "earlystop_snr": None,
    "relax_delay": 60.0,
    "reps": 3000,
    "rounds": 3,
}


def _pi_pulse(ml, freq: float, sim_params):
    return calibrated_drive_pulse(ml, "t1_pi", freq, sim_params=sim_params)


def test_t1_acquire_fits_finite_positive_t1():
    ctrl = build_core()
    sim_params = high_snr_simparams(20_000.0)
    connect_mock(ctrl, sim_params=sim_params)
    ml = ctrl.state.exp_context.ml
    predictor = mock_flux_predictor(sim_params)

    builder = T1Builder()
    schema = node_schema(builder, _PARAMS)
    flux_values = [0.0, 0.1]
    t1s: list[float] = []
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(schema, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, schema=schema, ml=ml, result=result
        )
        f01 = float(predictor.predict_freq(flux))
        snap = Snapshot(
            {"t1": 10.0},
            modules={
                "pi_pulse": _pi_pulse(ml, f01, sim_params),
                "opt_readout": ACQUIRE_READOUT,
            },
        )
        patch = builder.build_node(env).produce(snap)
        if "t1" in patch.values():
            t1s.append(float(patch.values()["t1"]))

    assert t1s, "no flux point fit a T1 from the real acquire"
    assert all(np.isfinite(t) and t > 0.0 for t in t1s), f"non-physical T1: {t1s}"
