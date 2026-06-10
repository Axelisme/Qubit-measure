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
