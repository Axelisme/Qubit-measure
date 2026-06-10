"""Phase B cfg-builder tests — a Node Builder lowers the active context + this
point's snapshot into a runnable base cfg (no acquire here, just construction).

The fixture builds a minimal ModuleLibrary (the qub drive waveform) + a raw
readout module, mirroring a real ``module_cfg.yaml``, so ``make_cfg`` exercises
the real ``ml.make_cfg`` lowering path without hardware.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import (
    QubitFreqBuilder,
    QubitFreqCfgTemplate,
)
from zcu_tools.meta_tool import ModuleLibrary

_READOUT = {
    "type": "readout/pulse",
    "pulse_cfg": {
        "ch": 0,
        "nqz": 2,
        "freq": 7444.6,
        "gain": 1.0,
        "waveform": {"style": "const", "length": 1.0},
    },
    "ro_cfg": {"ro_ch": 0, "ro_length": 0.9, "trig_offset": 0.6},
}


def _ml() -> ModuleLibrary:
    ml = ModuleLibrary()
    ml.register_waveform(
        qub_flat={
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "cosine", "length": 0.02},
        }
    )
    return ml


def _env(ml: ModuleLibrary) -> RunEnv:
    return RunEnv(
        flux=0.0,
        flux_idx=0,
        params={
            "qub_waveform": "qub_flat",
            "qub_ch": 3,
            "qub_nqz": 2,
            "qub_gain": 0.05,
            "reps": 100,
            "rounds": 2,
        },
        ml=ml,
    )


def test_qubit_freq_make_cfg_lowers_context():
    snap = Snapshot(
        {"predict_freq": 5135.0, "fit_kappa": 0.05}, modules={"readout": _READOUT}
    )
    cfg = QubitFreqBuilder().make_cfg(_env(_ml()), snap)
    assert isinstance(cfg, QubitFreqCfgTemplate)
    # the drive pulse frequency is the predicted qubit freq (from the snapshot)
    assert float(cfg.modules.qub_pulse.freq) == 5135.0
    assert int(cfg.modules.qub_pulse.ch) == 3
    assert cfg.reps == 100 and cfg.rounds == 2


def test_qubit_freq_produce_builds_cfg_when_context_configured():
    # with a populated ml + the drive params + a readout, produce goes through the
    # cfg pipeline (make_cfg) to source the centre freq, then SIMULATES the acquire
    # (no hardware). The pure-synthetic fallback (empty ml) is covered by the run
    # tests in test_run_body / test_feedback.
    import numpy as np
    from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import (
        QubitFreqNode,
        parse_detune_sweep,
    )
    from zcu_tools.gui.app.autofluxdep.nodes.result import QubitFreqResult
    from zcu_tools.gui.app.autofluxdep.tools import SimplePredictor, Tools

    ml = _ml()
    detune = parse_detune_sweep("-20,50,0.5")
    result = QubitFreqResult.allocate(np.array([0.0]), detune)
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params={
            "qub_waveform": "qub_flat",
            "qub_ch": 3,
            "qub_nqz": 2,
            "qub_gain": 0.05,
            "rounds": 2,
            "acquire_delay": 0,
        },
        ml=ml,
        result=result,
        tools=Tools(predictor=SimplePredictor()),
    )
    snap = Snapshot(
        {"predict_freq": 5135.0, "fit_kappa": 0.05}, modules={"readout": _READOUT}
    )
    node = QubitFreqBuilder().build_node(env)
    assert isinstance(node, QubitFreqNode)
    assert node._maybe_make_cfg(snap) is not None  # cfg-driven path is taken
    patch = node.produce(snap)
    assert "qubit_freq" in patch.values()  # simulated acquire + fit succeeded


def test_lenrabi_make_cfg_lowers_context():
    from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import (
        LenRabiBuilder,
        LenRabiCfgTemplate,
    )

    ml = _ml()
    params = {
        "qub_waveform": "qub_flat",
        "qub_ch": 4,
        "qub_nqz": 2,
        "qub_gain": 0.3,
        "qub_length": 0.5,
        "sweep_range": "0.05,2.5,61",
        "reps": 1000,
        "rounds": 10,
        "relax_delay": 1.0,
    }
    env = RunEnv(flux=0.0, flux_idx=0, params=params, ml=ml)
    snap = Snapshot({"qubit_freq": 5135.0}, modules={"opt_readout": _READOUT})
    cfg = LenRabiBuilder().make_cfg(env, snap)
    assert isinstance(cfg, LenRabiCfgTemplate)
    # the rabi drive pulse is on resonance: its freq is the required qubit_freq
    assert float(cfg.modules.rabi_pulse.freq) == 5135.0
    assert int(cfg.modules.rabi_pulse.ch) == 4
    assert cfg.reps == 1000 and cfg.rounds == 10
    # sweep_range is the (start, stop) extent parsed from the param axis
    assert cfg.sweep_range == (0.05, 2.5)


def test_lenrabi_produce_builds_cfg_when_context_configured():
    # with a populated ml + the drive params + a readout, produce goes through the
    # cfg pipeline (make_cfg), then SIMULATES the acquire (no hardware). The
    # pure-synthetic fallback (empty ml) is covered by test_builders.
    import numpy as np
    from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import (
        LenRabiBuilder,
        LenRabiNode,
    )
    from zcu_tools.gui.app.autofluxdep.nodes.synth import parse_linear_axis

    ml = _ml()
    params = {
        "qub_waveform": "qub_flat",
        "qub_ch": 4,
        "qub_nqz": 2,
        "qub_gain": 0.3,
        "qub_length": 0.5,
        "sweep_range": "0.05,2.5,61",
        "rounds": 2,
        "acquire_delay": 0,
    }
    flux_arr = np.linspace(0.0, 1.0, 11)  # idx=1 → flux=0.1 (high SNR, fittable)
    builder = LenRabiBuilder()
    result = builder.make_init_result(params, flux_arr)
    env = RunEnv(
        flux=float(flux_arr[1]),
        flux_idx=1,
        params=params,
        ml=ml,
        result=result,
    )
    snap = Snapshot({"qubit_freq": 5135.0}, modules={"opt_readout": _READOUT})
    node = builder.build_node(env)
    assert isinstance(node, LenRabiNode)
    # cfg-driven path is taken, and its sweep extent matches the Result axis
    cfg = node._maybe_make_cfg(snap)
    assert cfg is not None
    xs = parse_linear_axis(params["sweep_range"], (0.0, 6.0, 121))
    assert cfg.sweep_range == (float(xs[0]), float(xs[-1]))
    patch = node.produce(snap)
    assert "pi_length" in patch.values()  # simulated acquire + fit succeeded


def test_ro_optimize_make_cfg_lowers_context():
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
    from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import (
        RoOptimizeBuilder,
        RoOptimizeCfgTemplate,
    )
    from zcu_tools.meta_tool import ModuleLibrary

    ml = ModuleLibrary()
    ml.register_waveform(
        qub_flat={
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "cosine", "length": 0.02},
        }
    )
    # the pi_pulse module ro_optimize lowers comes whole from the snapshot
    # (lenrabi produces it upstream); a concrete PulseCfg raw dict here.
    pi_pulse = {
        "type": "pulse",
        "waveform": ml.get_waveform("qub_flat", {"length": 0.1}),
        "ch": 3,
        "nqz": 2,
        "gain": 0.3,
        "freq": 5135.0,
    }
    snap = Snapshot(
        {"best_ro_freq": 7444.6, "best_ro_gain": 0.5, "t1": 10.0},
        modules={"pi_pulse": pi_pulse, "readout": _READOUT},
    )
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params={
            "reps": 1000,
            "rounds": 10,
            "freq_window": 1.0,
            "gain_window": 0.05,
        },
        ml=ml,
    )
    cfg = RoOptimizeBuilder().make_cfg(env, snap)
    assert isinstance(cfg, RoOptimizeCfgTemplate)
    # the sweep windows are centred on the previous best ± the window half-widths
    assert cfg.freq_range == (7443.6, 7445.6)
    assert cfg.gain_range == (0.45, 0.55)
    # the readout the cfg sweeps over comes whole from the snapshot
    assert float(cfg.modules.readout.pulse_cfg.freq) == 7444.6
    assert cfg.reps == 1000 and cfg.rounds == 10
    # relax_delay = 3 * (smoothed) T1
    assert cfg.relax_delay == 30.0


def test_ro_optimize_produce_builds_cfg_when_context_configured():
    # with a populated ml + the pi_pulse / readout modules, produce goes through
    # the cfg pipeline (make_cfg) to source the plant-centre freq / gain, then
    # SIMULATES the acquire (no hardware). The pure-synthetic fallback (ml None /
    # readout None) is covered by the run + builder tests.
    import numpy as np
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
    from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import (
        RoOptimizeBuilder,
        RoOptimizeNode,
    )
    from zcu_tools.meta_tool import ModuleLibrary

    ml = ModuleLibrary()
    ml.register_waveform(
        qub_flat={
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "cosine", "length": 0.02},
        }
    )
    pi_pulse = {
        "type": "pulse",
        "waveform": ml.get_waveform("qub_flat", {"length": 0.1}),
        "ch": 3,
        "nqz": 2,
        "gain": 0.3,
        "freq": 5135.0,
    }
    flux_arr = np.linspace(0.0, 1.0, 11)
    params = {
        "freq_expts": 21,
        "gain_expts": 21,
        "reps": 1000,
        "rounds": 2,
        "acquire_delay": 0,
        "freq_range": (7443.0, 7446.0, 21),
        "gain_range": (0.4, 0.6, 21),
        "freq_window": 1.0,
        "gain_window": 0.05,
    }
    result = RoOptimizeBuilder().make_init_result(params, flux_arr)
    env = RunEnv(
        flux=float(flux_arr[1]),
        flux_idx=1,
        params=params,
        ml=ml,
        result=result,
    )
    snap = Snapshot(
        {"best_ro_freq": 7444.6, "best_ro_gain": 0.5, "t1": 10.0},
        modules={"pi_pulse": pi_pulse, "readout": _READOUT},
    )
    node = RoOptimizeBuilder().build_node(env)
    assert isinstance(node, RoOptimizeNode)
    assert node._maybe_make_cfg(snap) is not None  # cfg-driven path is taken
    patch = node.produce(snap)
    # simulated acquire + argmax produced the best point + the tuned readout module
    assert "best_ro_freq" in patch.values() and "best_ro_gain" in patch.values()
    assert "opt_readout" in patch.modules()
    assert not np.isnan(result.signal[1]).any()  # row filled
    assert not np.isnan(result.best_freq[1])
