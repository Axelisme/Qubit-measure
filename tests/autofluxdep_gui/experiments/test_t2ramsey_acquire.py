"""RB-2: t2ramsey end-to-end real acquire against the flux-aware MockSoc.

Runs the t2ramsey Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(Reset, pi/2, Delay, detuned pi/2, Readout) over the delay sweep ->
fit_decay_fringe). The current mock Ramsey fringe is measured and preserved as raw
evidence, but it does not clear the legacy feedback-success gate, so no T2Ramsey
Patch keys are emitted.

The pi/2 pulses drive at the predicted f01; an activate-detune phase ramp on the
second pulse makes the fringe resolvable (lower-layer behaviour).
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.experiments.t2ramsey import T2RamseyBuilder
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.cfg import SweepValue
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from .._helpers import (
    ACQUIRE_READOUT,
    connect_mock,
    high_snr_simparams,
    make_acquire_env,
    node_schema,
)

# The stricter legacy feedback gate needs a clean mock fringe; high_snr_simparams
# keeps this as a real acquire/fit test without paying extra rounds.
# sweep_range overrides the schema default's 121 pts → 61 pts for test speed;
# the production default stays at 121 in the node schema.
_PARAMS = {
    "reps": 1000,
    "rounds": 1,
    "detune_ratio": 0.2,
    "sweep_range": SweepValue(start=0.0, stop=25.0, expts=61),
}


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


def test_t2ramsey_acquire_rejects_untrusted_fit_but_keeps_raw_row():
    ctrl = build_core()
    connect_mock(ctrl, sim_params=high_snr_simparams())
    ml = ctrl.state.exp_context.ml
    predictor = FluxoniumPredictor(
        params=(4.0, 1.0, 1.0), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )

    builder = T2RamseyBuilder()
    schema = node_schema(builder, _PARAMS)
    flux_values = [0.0, 0.1]
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(schema, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, schema=schema, ml=ml, result=result
        )
        f01 = float(predictor.predict_freq(flux))
        snap = Snapshot(
            {"t1": 10.0, "t2r": 8.0},
            modules={"pi2_pulse": _pi2_pulse(ml, f01), "opt_readout": ACQUIRE_READOUT},
        )
        patch = builder.build_node(env).produce(snap)

        assert "t2r" not in patch.values()
        assert "t2r_detune" not in patch.values()
        assert not np.isnan(result.signal[idx]).all()
        assert np.isnan(result.fit_value[idx])
        assert np.isnan(result.fit_curve[idx]).all()
