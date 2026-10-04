"""Exercise MIST acquisition through the flux-aware MockSoc.

The simulator verifies that the acquisition path fills a finite disturbance row.
It does not validate the physical high-gain punch-out model. Acquisition errors
remain test failures rather than being classified as simulator limitations.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.experiment.v2_gui.autofluxdep.mist import MistBuilder
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.cfg import SweepValue
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from tests.gui.app.autofluxdep._helpers import (
    ACQUIRE_READOUT,
    connect_mock,
    make_acquire_env,
    node_schema,
)
from tests.gui.app.autofluxdep._helpers import build_test_core as build_core

_PARAMS = {
    "gain_sweep": SweepValue(start=0.0, stop=1.0, expts=21),
    "mist_ch": 1,
    "mist_nqz": 1,
    "mist_freq": 0.0,
    "mist_gain": 0.5,
    "mist_length": 0.1,
    "reps": 100,
    "rounds": 1,
    "relax_delay": 0.0,
}


def _pi_pulse(ml, freq: float):
    ml.register_waveform(mist_drive={"style": "const", "length": 1.0})
    return {
        "type": "pulse",
        "waveform": ml.get_waveform("mist_drive", {"length": 0.1}),
        "ch": 1,
        "nqz": 1,
        "gain": 0.5,
        "freq": freq,
    }


def test_mist_acquire_reports_success_and_finite_signal():
    ctrl = build_core()
    connect_mock(ctrl)
    ml = ctrl.state.session_env.ml
    predictor = FluxoniumPredictor(
        params=(4.0, 1.0, 1.0), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )

    builder = MistBuilder()
    flux = 0.0
    f01 = float(predictor.predict_freq(flux))
    # the mist_freq knob sets the disturbance drive; set it on resonance so the
    # mist pulse actually drives the qubit under the SimEngine (before building
    # the schema the env lowers).
    params = {**_PARAMS, "mist_freq": f01}
    schema = node_schema(builder, params)
    result = builder.make_init_result(schema, np.asarray([flux]))
    env = make_acquire_env(
        ctrl, flux=flux, flux_idx=0, schema=schema, ml=ml, result=result
    )
    snap = Snapshot(
        {"success": 1.0},
        modules={"pi_pulse": _pi_pulse(ml, f01), "opt_readout": ACQUIRE_READOUT},
    )

    patch = builder.build_node(env).produce(snap)

    # the real acquire ran: success reported + the disturbance row is finite
    assert patch.values().get("success") == 1.0
    assert np.all(np.isfinite(result.signal[0]))
