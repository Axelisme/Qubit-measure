"""RB-2: ro_optimize end-to-end real acquire against the flux-aware MockSoc.

Runs the ro_optimize Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(Reset, ge-Branch(pi_pulse), PulseReadout) over the freq x gain
sweep with a MomentTracker -> snr_as_signal -> argmax) and asserts a finite best
(freq, gain) inside the sweep window comes back, and the SNR landscape row is
filled (not all-nan). No fit -- the argmax of the SNR map is the result.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder

from ._helpers import connect_mock, make_acquire_env

# the swept readout pulse template (its freq/gain are swept by the node)
_READOUT = {
    "type": "readout/pulse",
    "pulse_cfg": {
        "ch": 0,
        "nqz": 2,
        "freq": 6000.0,
        "gain": 0.5,
        "waveform": {"style": "const", "length": 1.0},
    },
    "ro_cfg": {"ro_ch": 0, "ro_length": 0.9, "trig_offset": 0.6},
}

_PARAMS = {
    "freq_expts": 11,
    "gain_expts": 11,
    "reps": 100,
    "rounds": 1,
    "freq_window": 5.0,
    "gain_window": 0.3,
}


def _pi_pulse(ml):
    ml.register_waveform(ro_pi={"style": "const", "length": 1.0})
    return {
        "type": "pulse",
        "waveform": ml.get_waveform("ro_pi", {"length": 0.1}),
        "ch": 1,
        "nqz": 1,
        "gain": 0.3,
        "freq": 600.0,
    }


def test_ro_optimize_acquire_finds_best_point():
    ctrl = build_core()
    connect_mock(ctrl)
    ml = ctrl.state.exp_context.ml
    pi_pulse = _pi_pulse(ml)

    builder = RoOptimizeBuilder()
    flux_values = [0.0, 0.1]
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(_PARAMS, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, params=_PARAMS, ml=ml, result=result
        )
        snap = Snapshot(
            {"best_ro_freq": 6000.0, "best_ro_gain": 0.5, "t1": 10.0},
            modules={"pi_pulse": pi_pulse, "readout": _READOUT},
        )
        patch = builder.build_node(env).produce(snap)

        best_freq = patch.values()["best_ro_freq"]
        best_gain = patch.values()["best_ro_gain"]
        # finite best point inside the swept window
        assert np.isfinite(best_freq) and np.isfinite(best_gain)
        assert result.freq[0] <= best_freq <= result.freq[-1]
        assert result.gain[0] <= best_gain <= result.gain[-1]
        # the SNR landscape row was filled (real acquire produced data)
        assert not np.isnan(result.signal[idx]).all()
        # a tuned readout module flows downstream
        assert "opt_readout" in patch.modules()
