"""RB-2: ro_optimize end-to-end real acquire against the flux-aware MockSoc.

Runs the ro_optimize Node's real acquire path (set flux device -> setup_devices ->
ModularProgramV2(ge-Branch(pi_pulse), PulseReadout) over the freq x gain
sweep with a MomentTracker -> snr_as_signal -> argmax) and asserts a finite best
(freq, gain) inside the sweep window comes back, and the SNR landscape row is
filled (not all-nan). No fit -- the argmax of the SNR map is the result.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder
from zcu_tools.gui.cfg import CenteredSweepValue

from ._helpers import (
    calibrated_drive_pulse,
    connect_mock,
    high_snr_simparams,
    make_acquire_env,
    mock_flux_predictor,
    node_schema,
)

# the swept readout pulse template (its freq/gain are swept by the node)
_READOUT = {
    "type": "readout/pulse",
    "pulse_cfg": {
        "ch": 0,
        "nqz": 2,
        "freq": 6000.0,
        "gain": 0.3,
        "waveform": {"style": "const", "length": 1.0},
    },
    "ro_cfg": {"ro_ch": 0, "ro_freq": 6000.0, "ro_length": 0.9, "trig_offset": 0.6},
}

_PARAMS = {
    "reps": 2000,
    "rounds": 8,
    "freq_range": CenteredSweepValue(center=6000.0, span=10.0, expts=11),
    "gain_range": CenteredSweepValue(center=0.5, span=0.6, expts=11),
    "relax_delay_mode": "fixed",
    "relax_delay": 60.0,
    "skew_penalty": 0.0,
}


def _pi_pulse(ml, sim_params):
    freq = float(mock_flux_predictor(sim_params).predict_freq(0.0))
    return calibrated_drive_pulse(ml, "ro_pi", freq, sim_params=sim_params)


def test_ro_optimize_acquire_finds_best_point():
    ctrl = build_core()
    sim_params = high_snr_simparams(20_000.0)
    connect_mock(ctrl, sim_params=sim_params)
    ml = ctrl.state.exp_context.ml
    pi_pulse = _pi_pulse(ml, sim_params)

    builder = RoOptimizeBuilder()
    schema = node_schema(builder, _PARAMS)
    flux_values = [0.0, 0.1]
    for idx, flux in enumerate(flux_values):
        result = builder.make_init_result(schema, np.asarray(flux_values))
        env = make_acquire_env(
            ctrl, flux=flux, flux_idx=idx, schema=schema, ml=ml, result=result
        )
        snap = Snapshot(
            {"best_ro_freq": 6000.0, "best_ro_gain": 0.5, "t1": 10.0},
            modules={"pi_pulse": pi_pulse, "readout": _READOUT},
        )
        patch = builder.build_node(env).produce(snap)

        best_freq = patch.values()["best_ro_freq"]
        best_gain = patch.values()["best_ro_gain"]
        # produce derives the actual swept axes from make_cfg(), not from stale
        # allocation defaults. Flux point 0 uses the editable initial window;
        # later points use the generated previous-best window around snap.
        expected_freq_edges = [5995.0, 6005.0] if idx == 0 else [5999.0, 6001.0]
        expected_gain_edges = [0.2, 0.8] if idx == 0 else [0.4, 0.6]
        np.testing.assert_allclose(result.freq[[0, -1]], expected_freq_edges)
        np.testing.assert_allclose(result.gain[[0, -1]], expected_gain_edges)
        # finite best point inside the swept window
        assert np.isfinite(best_freq) and np.isfinite(best_gain)
        assert result.freq[0] <= best_freq <= result.freq[-1]
        assert result.gain[0] <= best_gain <= result.gain[-1]
        # the SNR landscape row was filled (real acquire produced data)
        assert not np.isnan(result.signal[idx]).all()
        # a tuned readout module flows downstream
        assert "opt_readout" in patch.modules()


def test_ro_optimize_acquire_leaves_cooperative_stop_to_schedule(monkeypatch):
    ctrl = build_core()
    sim_params = high_snr_simparams()
    connect_mock(ctrl, sim_params=sim_params)
    ml = ctrl.state.exp_context.ml
    pi_pulse = _pi_pulse(ml, sim_params)

    captured: dict[str, object] = {}

    class FakeProgram:
        def __init__(self, _soccfg, cfg, *, modules, sweep):
            del modules, sweep
            self.cfg_model = cfg

        def acquire(self, *args, **kwargs):
            del args
            cancel_flag = kwargs["cancel_flag"]
            captured["cancel_flag_initial"] = cancel_flag.is_set()
            captured["trackers"] = kwargs.get("trackers")
            kwargs["round_hook"](1, object(), cancel_flag)
            captured["cancel_flag_after_round"] = cancel_flag.is_set()
            return object()

        def acquire_decimated(self, *args, **kwargs):
            del args, kwargs
            raise NotImplementedError

    def fake_landscape(_tracker, shape, *, skew_penalty):
        del _tracker, skew_penalty
        values = np.zeros(shape, dtype=float)
        values[0, 0] = 1.0
        return values

    from zcu_tools.gui.app.autofluxdep.nodes import ro_optimize as ro_mod

    monkeypatch.setattr(ro_mod, "ModularProgramV2", FakeProgram)
    monkeypatch.setattr(ro_mod, "_ro_landscape", fake_landscape)

    builder = RoOptimizeBuilder()
    schema = node_schema(builder, {**_PARAMS, "rounds": 2})
    result = builder.make_init_result(schema, np.asarray([0.0]))

    stop_polls = {"count": 0}

    def should_stop() -> bool:
        stop_polls["count"] += 1
        return True

    env = make_acquire_env(
        ctrl,
        flux=0.0,
        flux_idx=0,
        schema=schema,
        ml=ml,
        result=result,
        should_stop=should_stop,
    )
    snap = Snapshot(
        {"best_ro_freq": 6000.0, "best_ro_gain": 0.5, "t1": 10.0},
        modules={"pi_pulse": pi_pulse, "readout": _READOUT},
    )

    builder.build_node(env).produce(snap)

    assert captured["cancel_flag_initial"] is False
    assert captured["cancel_flag_after_round"] is False
    assert stop_polls["count"] == 0
    trackers = captured["trackers"]
    assert isinstance(trackers, list)
    assert trackers
