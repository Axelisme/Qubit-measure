"""Phase B cfg-builder tests — a Node Builder lowers the active context + this
point's snapshot into a runnable base cfg (no acquire here, just construction).

The fixture builds a minimal ModuleLibrary (the qub drive waveform) + a raw
readout module, mirroring a real ``module_cfg.yaml``, so ``make_cfg`` exercises
the real ``ml.make_cfg`` lowering path without hardware.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.cfg import SweepValue
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


def test_qubit_freq_produce_fast_fails_when_context_unconfigured():
    # the real-acquire contract: produce Fast Fails (no synthetic fallback) when the
    # context is unconfigured — here ml is None, so make_cfg cannot lower a drive
    # pulse. The error must be clear; the orchestrator turns it into RUN_FAILED.
    import numpy as np
    import pytest

    builder = QubitFreqBuilder()
    result = builder.make_init_result(
        {"detune_sweep": SweepValue(start=-20.0, stop=50.0, expts=141)},
        np.array([0.0]),
    )
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params={"qub_waveform": "qub_flat", "qub_ch": 3},
        ml=None,
        result=result,
    )
    snap = Snapshot(
        {"predict_freq": 5135.0, "fit_kappa": 0.05}, modules={"readout": _READOUT}
    )
    with pytest.raises(RuntimeError, match="ModuleLibrary"):
        builder.build_node(env).produce(snap)


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
        "sweep_range": SweepValue(start=0.05, stop=2.5, expts=61),
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


def test_lenrabi_produce_fast_fails_when_context_unconfigured():
    # the real-acquire contract: produce Fast Fails (no synthetic fallback) when ml
    # is None — make_cfg cannot lower the rabi drive pulse.
    import numpy as np
    import pytest
    from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder

    builder = LenRabiBuilder()
    params = {
        "qub_waveform": "qub_flat",
        "qub_ch": 4,
        "sweep_range": SweepValue(start=0.05, stop=2.5, expts=61),
    }
    result = builder.make_init_result(params, np.linspace(0.0, 1.0, 11))
    env = RunEnv(flux=0.1, flux_idx=1, params=params, ml=None, result=result)
    snap = Snapshot({"qubit_freq": 5135.0}, modules={"opt_readout": _READOUT})
    with pytest.raises(RuntimeError, match="ModuleLibrary"):
        builder.build_node(env).produce(snap)


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


def test_ro_optimize_produce_fast_fails_when_context_unconfigured():
    # the real-acquire contract: produce Fast Fails (no synthetic fallback) when ml
    # is None — make_cfg cannot lower the swept readout pulse.
    import numpy as np
    import pytest
    from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder

    builder = RoOptimizeBuilder()
    params = {
        "freq_expts": 21,
        "gain_expts": 21,
        "freq_window": 1.0,
        "gain_window": 0.05,
    }
    result = builder.make_init_result(params, np.linspace(0.0, 1.0, 11))
    env = RunEnv(flux=0.1, flux_idx=1, params=params, ml=None, result=result)
    snap = Snapshot(
        {"best_ro_freq": 7444.6, "best_ro_gain": 0.5, "t1": 10.0},
        modules={"pi_pulse": _T1_PI_PULSE, "readout": _READOUT},
    )
    with pytest.raises(RuntimeError, match="ModuleLibrary"):
        builder.build_node(env).produce(snap)


# --- t1 (Phase B B-1/B-2) ----------------------------------------------------
# t1's drive pi_pulse + readout are MODULES taken from the snapshot (lenrabi /
# ro_optimize produce them), not built from "設定頭" params; relax_delay +
# sweep_range derive from the snapshot's smoothed t1. The configured fixture
# supplies real PulseCfg / PulseReadoutCfg module dicts so make_cfg exercises the
# real ml.make_cfg lowering path without hardware. (Reuses the module-level
# _READOUT fixture already defined at the top of test_cfg_maker.py.)

_T1_PI_PULSE = {
    "type": "pulse",
    "waveform": {
        "style": "flat_top",
        "length": 0.1,
        "raise_waveform": {"style": "cosine", "length": 0.02},
    },
    "ch": 3,
    "nqz": 2,
    "freq": 5135.0,
    "gain": 0.3,
}


def test_t1_make_cfg_lowers_context():
    from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
    from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
    from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder, T1CfgTemplate
    from zcu_tools.meta_tool import ModuleLibrary

    env = RunEnv(
        flux=0.0, flux_idx=0, params={"reps": 100, "rounds": 2}, ml=ModuleLibrary()
    )
    snap = Snapshot(
        {"t1": 12.0}, modules={"pi_pulse": _T1_PI_PULSE, "opt_readout": _READOUT}
    )
    cfg = T1Builder().make_cfg(env, snap)
    assert isinstance(cfg, T1CfgTemplate)
    # the drive pi_pulse + readout are the snapshot modules, lowered to real cfgs
    assert int(cfg.modules.pi_pulse.ch) == 3
    assert float(cfg.modules.pi_pulse.freq) == 5135.0
    assert cfg.modules.readout.type == "readout/pulse"
    # relax_delay + sweep_range derive from the smoothed t1 (the notebook formula)
    assert cfg.relax_delay == 36.0  # max(1.0, 3 * 12)
    assert cfg.sweep_range == (0.5, 60.0)  # (0.5, max(1.0, 5 * 12))
    # reps / rounds come from the node params
    assert cfg.reps == 100 and cfg.rounds == 2


def test_t1_produce_fast_fails_when_context_unconfigured():
    # the real-acquire contract: produce Fast Fails (no synthetic fallback) when ml
    # is None — make_cfg cannot lower the pi_pulse + readout.
    import numpy as np
    import pytest
    from zcu_tools.gui.app.autofluxdep.nodes.acquire import parse_linear_axis
    from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
    from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder

    times = parse_linear_axis("0.5,60,101", (0.5, 60.0, 101))
    result = Sweep1DResult.allocate(np.array([0.5]), times, x_label="relax time (us)")
    env = RunEnv(
        flux=0.5,
        flux_idx=0,
        params={
            "sweep_range": SweepValue(start=0.5, stop=60.0, expts=101),
            "reps": 100,
            "rounds": 2,
        },
        ml=None,
        result=result,
    )
    snap = Snapshot(
        {"t1": 12.0}, modules={"pi_pulse": _T1_PI_PULSE, "opt_readout": _READOUT}
    )
    with pytest.raises(RuntimeError, match="ModuleLibrary"):
        T1Builder().build_node(env).produce(snap)


def _t2ramsey_ml() -> ModuleLibrary:
    ml = ModuleLibrary()
    ml.register_waveform(
        pi2_flat={
            "style": "flat_top",
            "length": 0.05,
            "raise_waveform": {"style": "cosine", "length": 0.01},
        }
    )
    return ml


def _t2ramsey_pi2_pulse(ml: ModuleLibrary):
    # the fully-built pi/2 drive module lenrabi would produce (here built off the
    # registered waveform), passed straight into make_cfg's pi2_pulse slot.
    return {
        "type": "pulse",
        "waveform": ml.get_waveform("pi2_flat"),
        "ch": 3,
        "nqz": 2,
        "gain": 0.3,
        "freq": 5135.0,
    }


def test_t2ramsey_make_cfg_lowers_context():
    from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import (
        T2RamseyBuilder,
        T2RamseyCfgTemplate,
    )

    ml = _t2ramsey_ml()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        params={"reps": 1000, "rounds": 10},
        ml=ml,
    )
    snap = Snapshot(
        {"t1": 12.0, "t2r": 8.0},
        modules={"pi2_pulse": _t2ramsey_pi2_pulse(ml), "opt_readout": _READOUT},
    )
    cfg = T2RamseyBuilder().make_cfg(env, snap)
    assert isinstance(cfg, T2RamseyCfgTemplate)
    # the pi/2 drive module is taken from the snapshot
    assert int(cfg.modules.pi2_pulse.ch) == 3
    assert float(cfg.modules.pi2_pulse.freq) == 5135.0
    # sweep_range spans 2.5 * smoothed t2r; relax_delay is 3 * smoothed t1
    assert cfg.sweep_range == (0.0, 2.5 * 8.0)
    assert cfg.relax_delay == max(1.0, 3.0 * 12.0)
    assert cfg.reps == 1000 and cfg.rounds == 10
    # the readout module (snapshot's opt_readout) lowered into the cfg's readout
    assert cfg.modules.readout.type == "readout/pulse"


def test_t2ramsey_produce_fast_fails_when_context_unconfigured():
    # the real-acquire contract: produce Fast Fails (no synthetic fallback) when ml
    # is None — make_cfg cannot lower the pi/2 pulse + readout.
    import numpy as np
    import pytest
    from zcu_tools.gui.app.autofluxdep.nodes.acquire import parse_linear_axis
    from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
    from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder

    flux = np.linspace(0.0, 1.0, 11)
    times = parse_linear_axis("0,25,61", (0.0, 25.0, 61))
    result = Sweep1DResult.allocate(flux, times, x_label="delay time (us)")
    env = RunEnv(
        flux=float(flux[1]),
        flux_idx=1,
        params={"reps": 1000, "rounds": 2},
        ml=None,
        result=result,
    )
    snap = Snapshot(
        {"t1": 12.0, "t2r": 8.0},
        modules={
            "pi2_pulse": _t2ramsey_pi2_pulse(_t2ramsey_ml()),
            "opt_readout": _READOUT,
        },
    )
    with pytest.raises(RuntimeError, match="ModuleLibrary"):
        T2RamseyBuilder().build_node(env).produce(snap)


# --- t2echo -----------------------------------------------------------------


def _t2echo_pulses(ml: ModuleLibrary):
    """Real PulseCfg dicts for the pi / pi2 drive pulses (lenrabi-shaped output).

    The lower-layer T2EchoModuleCfg requires concrete PulseCfg modules (type /
    waveform / ch / nqz / freq / gain), unlike the synthetic-path sentinels used
    in the run tests; build them off the fixture's registered waveform.
    """
    pi_pulse = {
        "type": "pulse",
        "waveform": ml.get_waveform("qub_flat", {"length": 0.1}),
        "ch": 3,
        "nqz": 2,
        "gain": 0.5,
        "freq": 5135.0,
    }
    pi2_pulse = {
        "type": "pulse",
        "waveform": ml.get_waveform("qub_flat", {"length": 0.05}),
        "ch": 3,
        "nqz": 2,
        "gain": 0.25,
        "freq": 5135.0,
    }
    return pi_pulse, pi2_pulse


def _t2echo_env(ml: ModuleLibrary) -> RunEnv:
    return RunEnv(
        flux=0.0,
        flux_idx=0,
        params={"reps": 1000, "rounds": 10},
        ml=ml,
    )


def test_t2echo_make_cfg_lowers_context():
    from zcu_tools.gui.app.autofluxdep.nodes.t2echo import (
        T2EchoBuilder,
        T2EchoCfgTemplate,
    )

    ml = _ml()
    pi_pulse, pi2_pulse = _t2echo_pulses(ml)
    snap = Snapshot(
        {"t1": 12.0, "t2e": 8.0},
        modules={
            "pi_pulse": pi_pulse,
            "pi2_pulse": pi2_pulse,
            "opt_readout": _READOUT,
        },
    )
    cfg = T2EchoBuilder().make_cfg(_t2echo_env(ml), snap)
    assert isinstance(cfg, T2EchoCfgTemplate)
    # the delay window is (0, 2.5 * smoothed_t2e) from the snapshot
    assert cfg.sweep_range == (0.0, 2.5 * 8.0)
    # relax_delay = max(1.0, 3 * smoothed_t1)
    assert float(cfg.relax_delay) == 3.0 * 12.0
    # the drive pulses come from the snapshot modules
    assert int(cfg.modules.pi_pulse.ch) == 3
    assert float(cfg.modules.pi_pulse.gain) == 0.5
    assert float(cfg.modules.pi2_pulse.gain) == 0.25
    assert cfg.modules.readout.type == "readout/pulse"
    assert cfg.reps == 1000 and cfg.rounds == 10


def test_t2echo_produce_fast_fails_when_context_unconfigured():
    # the real-acquire contract: produce Fast Fails (no synthetic fallback) when ml
    # is None — make_cfg cannot lower the pi / pi2 drive pulses + readout.
    import numpy as np
    import pytest
    from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult
    from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder

    pi_pulse, pi2_pulse = _t2echo_pulses(_ml())
    flux_arr = np.linspace(0.0, 1.0, 11)
    result = Sweep1DResult.allocate(
        flux_arr, np.linspace(0.0, 25.0, 61), x_label="delay time (us)"
    )
    env = RunEnv(
        flux=float(flux_arr[1]),
        flux_idx=1,
        params={"reps": 1000, "rounds": 1},
        ml=None,
        result=result,
    )
    snap = Snapshot(
        {"t1": 10.0, "t2e": 8.0},
        modules={
            "pi_pulse": pi_pulse,
            "pi2_pulse": pi2_pulse,
            "opt_readout": _READOUT,
        },
    )
    with pytest.raises(RuntimeError, match="ModuleLibrary"):
        T2EchoBuilder().build_node(env).produce(snap)


# --- mist (1D gain sweep, no fit; cfg drives the disturbance onset gain) ---

# a concrete pi_pulse module (PulseCfg shape): excited-state preparation pulse
_PI_PULSE = {
    "type": "pulse",
    "ch": 3,
    "nqz": 2,
    "freq": 5135.0,
    "gain": 0.4,
    "waveform": {"style": "const", "length": 0.05},
}


def _mist_ml() -> ModuleLibrary:
    ml = ModuleLibrary()
    ml.register_waveform(
        mist_flat={
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "cosine", "length": 0.02},
        }
    )
    return ml


def _mist_env(ml: ModuleLibrary, **result_tools) -> RunEnv:
    return RunEnv(
        flux=0.0,
        flux_idx=0,
        params={
            "mist_waveform": "mist_flat",
            "mist_ch": 4,
            "mist_nqz": 2,
            "mist_freq": 5135.0,
            "mist_gain": 0.5,
            "reps": 100,
            "rounds": 2,
        },
        ml=ml,
        **result_tools,
    )


def test_mist_make_cfg_lowers_context():
    from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder, MistCfgTemplate

    snap = Snapshot(
        {"success": 1.0},
        modules={"pi_pulse": _PI_PULSE, "opt_readout": _READOUT},
    )
    cfg = MistBuilder().make_cfg(_mist_env(_mist_ml()), snap)
    assert isinstance(cfg, MistCfgTemplate)
    # the disturbance pulse channel / gain / freq come from the node params
    assert int(cfg.modules.mist_pulse.ch) == 4
    assert float(cfg.modules.mist_pulse.gain) == 0.5
    assert float(cfg.modules.mist_pulse.freq) == 5135.0
    # pi_pulse + readout were lowered from the snapshot modules
    assert int(cfg.modules.pi_pulse.ch) == 3
    assert cfg.modules.readout is not None
    assert cfg.reps == 100 and cfg.rounds == 2


def test_mist_produce_fast_fails_when_context_unconfigured():
    # the real-acquire contract: produce Fast Fails (no synthetic fallback) when ml
    # is None — make_cfg cannot lower the mist disturbance pulse + readout.
    import dataclasses

    import numpy as np
    import pytest
    from zcu_tools.gui.app.autofluxdep.nodes.acquire import parse_linear_axis
    from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder
    from zcu_tools.gui.app.autofluxdep.nodes.result import Sweep1DResult

    gains = parse_linear_axis("0,1,21", (0.0, 1.0, 21))
    result = Sweep1DResult.allocate(np.array([0.0]), gains, x_label="gain")
    # the mist env fixture always binds an ml; replace it with None for this test.
    env = dataclasses.replace(_mist_env(_mist_ml(), result=result), ml=None)
    snap = Snapshot(
        {"success": 1.0},
        modules={"pi_pulse": _PI_PULSE, "opt_readout": _READOUT},
    )
    with pytest.raises(RuntimeError, match="ModuleLibrary"):
        MistBuilder().build_node(env).produce(snap)
