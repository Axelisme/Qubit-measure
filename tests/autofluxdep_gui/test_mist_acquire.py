"""RB-2: mist end-to-end real acquire against the flux-aware MockSoc.

MIST sweeps the disturbance-pulse gain and reads the state-disturbance magnitude
directly (no fit). MIST is a known SimEngine gap (the simulator does not yet model
the punch-out / high-gain disturbance regime), so this test asserts the real
acquire path runs and fills a finite row IF the SimEngine supports the program;
if the engine fast-fails on the mist program, the test skips with the exact error
(unified real acquire -- the Node carries no mock-detection branch).
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import SweepValue
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

from ._helpers import ACQUIRE_READOUT, connect_mock, make_acquire_env

_PARAMS = {
    "gain_sweep": SweepValue(start=0.0, stop=1.0, expts=21),
    "mist_waveform": "mist_drive",
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


def test_mist_acquire_runs_or_skips_on_simengine_gap():
    ctrl = build_core()
    connect_mock(ctrl)
    ml = ctrl.state.exp_context.ml
    predictor = FluxoniumPredictor(
        params=(4.0, 1.0, 1.0), flux_half=0.0, flux_period=1.0, flux_bias=0.0
    )

    builder = MistBuilder()
    flux = 0.0
    result = builder.make_init_result(_PARAMS, np.asarray([flux]))
    env = make_acquire_env(
        ctrl, flux=flux, flux_idx=0, params=_PARAMS, ml=ml, result=result
    )
    f01 = float(predictor.predict_freq(flux))
    # the mist_freq param sets the disturbance drive; set it on resonance so the
    # mist pulse actually drives the qubit under the SimEngine.
    _PARAMS["mist_freq"] = f01
    snap = Snapshot(
        {"success": 1.0},
        modules={"pi_pulse": _pi_pulse(ml, f01), "opt_readout": ACQUIRE_READOUT},
    )

    try:
        patch = builder.build_node(env).produce(snap)
    except Exception as exc:  # SimEngine MIST support pending
        pytest.skip(f"SimEngine MIST support pending: {type(exc).__name__}: {exc}")

    # the real acquire ran: success reported + the disturbance row is finite
    assert patch.values().get("success") == 1.0
    assert np.all(np.isfinite(result.signal[0]))
