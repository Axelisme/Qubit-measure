"""Typed node-knob CfgSchema: structure, defaults, equivalence, seam.

Three families of test:

1. **Structure** — each node's ``make_default_schema`` declares exactly the user
   knobs (the typed node settings), and *no* derived/upstream field (predict_freq,
   relax=3·T1, the pi/readout modules) leaks into the spec.
2. **Defaults** — the lowered default knobs provide operator-ready initial values
   (notebook / measure-gui conventions for the pulse "設定頭"), and a node's
   ``make_cfg`` built from those defaults yields the expected run cfg
   (golden, field-by-field).
3. **Seam invariant** — only the ``cfg/`` seam (``__init__.py`` / ``schema.py`` /
   ``form.py``) may import ``zcu_tools.gui.app.main`` from inside the autofluxdep
   package.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
from zcu_tools.gui.app.autofluxdep.cfg import SweepSpec
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder
from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder
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

_PI_PULSE = {
    "type": "pulse",
    "ch": 3,
    "nqz": 2,
    "freq": 5135.0,
    "gain": 0.4,
    "waveform": {"style": "const", "length": 0.05},
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


# --- 1. structure: knob keys are the declared user knobs, no derived field ------

# Each node's exact user-knob key set (the typed knobs the schema owns). This is
# the golden structural contract: the spec declares these and nothing else (no
# derived/upstream field — see _DERIVED_FORBIDDEN below).
_EXPECTED_KEYS = {
    "qubit_freq": {
        "detune_sweep",
        "reps",
        "rounds",
        "relax_delay",
        "earlystop_snr",
        "qub_waveform",
        "qub_ch",
        "qub_nqz",
        "qub_gain",
        "qub_length",
    },
    "lenrabi": {
        "sweep_range",
        "reps",
        "rounds",
        "relax_delay",
        "earlystop_snr",
        "qub_waveform",
        "qub_ch",
        "qub_nqz",
        "qub_gain",
        "qub_length",
    },
    "ro_optimize": {
        "freq_expts",
        "gain_expts",
        "reps",
        "rounds",
        "freq_window",
        "gain_window",
    },
    "t1": {"sweep_range", "reps", "rounds", "earlystop_snr"},
    "t2ramsey": {"sweep_range", "detune_ratio", "reps", "rounds", "earlystop_snr"},
    "t2echo": {"sweep_range", "detune_ratio", "reps", "rounds", "earlystop_snr"},
    "mist": {
        "gain_sweep",
        "reps",
        "rounds",
        "relax_delay",
        "mist_waveform",
        "mist_ch",
        "mist_nqz",
        "mist_freq",
        "mist_gain",
        "mist_length",
    },
}

# The complete derived/upstream set that MUST NOT appear as a user knob (it is
# injected by make_cfg / produce from the predictor, prev-point fits, or modules).
_DERIVED_FORBIDDEN = {
    "predict_freq",
    "qubit_freq",
    "freq",
    "freq_range",
    "gain_range",
    "pi_pulse",
    "pi2_pulse",
    "readout",
    "opt_readout",
    "best_ro_freq",
    "best_ro_gain",
    "t1",
    "t2r",
    "t2e",
    "fit_kappa",
}


def test_schema_keys_match_declared_knobs():
    for builder in (
        QubitFreqBuilder(),
        LenRabiBuilder(),
        RoOptimizeBuilder(),
        T1Builder(),
        T2RamseyBuilder(),
        T2EchoBuilder(),
        MistBuilder(),
    ):
        schema = builder.make_default_schema()
        assert set(schema.keys) == _EXPECTED_KEYS[builder.name], builder.name


def test_no_derived_field_in_any_spec():
    for builder in (
        QubitFreqBuilder(),
        LenRabiBuilder(),
        RoOptimizeBuilder(),
        T1Builder(),
        T2RamseyBuilder(),
        T2EchoBuilder(),
        MistBuilder(),
    ):
        keys = set(builder.make_default_schema().keys)
        assert keys.isdisjoint(_DERIVED_FORBIDDEN), (builder.name, keys)


# --- 2. defaults: the lowered knobs provide operator-ready initial values -----


def test_qubit_freq_default_knobs():
    knobs = QubitFreqBuilder().make_default_schema().lower(None)
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 100
    assert knobs["relax_delay"] == 0.5
    assert knobs["earlystop_snr"] == 50.0
    assert knobs["qub_waveform"] == "qub_flat"
    assert knobs["qub_ch"] == 0
    assert knobs["qub_nqz"] == 2
    assert knobs["qub_gain"] == 0.05
    assert knobs["qub_length"] == 0.1
    # clearing an optional default still omits it (preserving the Fast-Fail guard)
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("qub_waveform", None)
    schema.set_field("qub_ch", None)
    schema.set_field("earlystop_snr", None)
    cleared = schema.lower(None)
    assert "qub_waveform" not in cleared
    assert "qub_ch" not in cleared
    assert "earlystop_snr" not in cleared
    # the detune sweep lowers to the prototype's (-20, 50, step 0.5) axis exactly
    detune = knobs["detune_sweep"]
    axis = np.linspace(float(detune.start), float(detune.stop), int(detune.expts))
    old = np.linspace(-20.0, 50.0, int(round((50 - (-20)) / 0.5)) + 1)
    assert np.allclose(axis, old)


def test_lenrabi_default_knobs():
    knobs = LenRabiBuilder().make_default_schema().lower(None)
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 10
    assert knobs["relax_delay"] == 30.0
    assert knobs["earlystop_snr"] == 30.0
    assert knobs["qub_waveform"] == "qub_flat"
    assert knobs["qub_ch"] == 0
    sweep = knobs["sweep_range"]
    assert np.allclose([float(sweep.start), float(sweep.stop)], [0.05, 0.5])
    assert int(sweep.expts) == 101


def test_mist_default_knobs():
    knobs = MistBuilder().make_default_schema().lower(None)
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 100
    assert knobs["relax_delay"] == 20.5
    assert knobs["mist_waveform"] == "mist_waveform"
    assert knobs["mist_ch"] == 0
    assert knobs["mist_nqz"] == 2
    assert knobs["mist_freq"] == 6000.0
    assert knobs["mist_gain"] == 0.5
    assert knobs["mist_length"] == 0.1
    gain = knobs["gain_sweep"]
    assert (float(gain.start), float(gain.stop), int(gain.expts)) == (0.0, 1.0, 51)


def test_ro_optimize_default_knobs():
    knobs = RoOptimizeBuilder().make_default_schema().lower(None)
    assert knobs["freq_expts"] == 21
    assert knobs["gain_expts"] == 21
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 10
    assert knobs["freq_window"] == 1.0
    assert knobs["gain_window"] == 0.05
    # the window half-widths are flat scalars, NOT a SweepSpec (decision)
    spec = RoOptimizeBuilder().make_default_schema().schema.spec
    assert not isinstance(spec.fields["freq_window"], SweepSpec)
    assert not isinstance(spec.fields["gain_window"], SweepSpec)


def test_t1_default_knobs():
    knobs = T1Builder().make_default_schema().lower(None)
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 10
    assert knobs["earlystop_snr"] == 20.0
    sweep = knobs["sweep_range"]
    assert (float(sweep.start), float(sweep.stop), int(sweep.expts)) == (0.5, 60.0, 101)


def test_t2_default_knobs():
    for builder in (T2RamseyBuilder(), T2EchoBuilder()):
        knobs = builder.make_default_schema().lower(None)
        assert knobs["reps"] == 1000
        assert knobs["rounds"] == 10
        assert knobs["detune_ratio"] == 0.05
        assert knobs["earlystop_snr"] == 20.0
        sweep = knobs["sweep_range"]
        assert (float(sweep.start), float(sweep.stop), int(sweep.expts)) == (
            0.0,
            25.0,
            121,
        ), builder.name


# --- 2b. equivalence: default-knob make_cfg == the prototype's hardcoded cfg ----
#
# The defaults live in the schema; building make_cfg from an empty placement (so
# every knob takes its schema default) must yield the expected run cfg, field by
# field. qubit_freq is the worked golden (TwoToneCfg with the drive "設定頭").


def test_qubit_freq_make_cfg_uses_schema_defaults():
    ml = _ml()
    builder = QubitFreqBuilder()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=builder.make_default_schema(),
        ml=ml,
    )
    snap = Snapshot(
        {"predict_freq": 5135.0, "fit_kappa": 0.05}, modules={"readout": _READOUT}
    )
    cfg = builder.make_cfg(env, snap)
    # the hardcoded prototype defaults, now sourced from the schema
    assert cfg.reps == 1000
    assert cfg.rounds == 100
    assert cfg.relax_delay == 0.5
    assert int(cfg.modules.qub_pulse.ch) == 0
    assert int(cfg.modules.qub_pulse.nqz) == 2
    assert float(cfg.modules.qub_pulse.gain) == 0.05
    assert float(cfg.modules.qub_pulse.freq) == 5135.0  # the injected predict_freq


def test_mist_make_cfg_uses_schema_defaults():
    ml = ModuleLibrary()
    ml.register_waveform(
        mist_waveform={
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "cosine", "length": 0.02},
        }
    )
    builder = MistBuilder()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=builder.make_default_schema(),
        ml=ml,
    )
    snap = Snapshot(
        {"success": 1.0}, modules={"pi_pulse": _PI_PULSE, "opt_readout": _READOUT}
    )
    cfg = builder.make_cfg(env, snap)
    assert cfg.reps == 1000
    assert cfg.rounds == 100
    assert cfg.relax_delay == 20.5
    assert int(cfg.modules.mist_pulse.ch) == 0
    assert float(cfg.modules.mist_pulse.gain) == 0.5
    assert float(cfg.modules.mist_pulse.freq) == 6000.0
    assert int(cfg.modules.mist_pulse.nqz) == 2


# --- 2c. set_field type coercion + unknown-key fast-fail (the 160a bridge) ------


def test_set_field_coerces_text_to_type():
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("reps", "250")  # text from the prototype's line-edit form
    schema.set_field("qub_gain", "0.2")
    knobs = schema.lower(None)
    assert knobs["reps"] == 250 and isinstance(knobs["reps"], int)
    assert knobs["qub_gain"] == 0.2 and isinstance(knobs["qub_gain"], float)


def test_set_field_unknown_key_fast_fails():
    import pytest

    schema = QubitFreqBuilder().make_default_schema()
    with pytest.raises(KeyError, match="Unknown node param"):
        schema.set_field("not_a_knob", 1)


def test_with_overrides_unknown_key_fast_fails():
    import pytest

    schema = T1Builder().make_default_schema()
    with pytest.raises(KeyError, match="Unknown node param"):
        schema.with_overrides({"bogus": 1})


# --- 2d. set_node_params (controller typed entry): type, sweep, fast-fail -------


def test_set_node_params_types_and_fast_fails():
    import pytest
    from zcu_tools.gui.app.autofluxdep.app import build_core
    from zcu_tools.gui.app.autofluxdep.cfg import SweepValue

    ctrl = build_core()
    node = ctrl.add_node_by_type("qubit_freq")
    index = ctrl.state.nodes.index(node)

    # scalar values are coerced to the declared types and written into the schema
    # SSOT (read back via the lowered knobs)
    ctrl.set_node_params(index, {"reps": "250", "qub_gain": "0.2", "qub_ch": "3"})
    knobs = node.schema.lower(None)
    assert knobs["reps"] == 250 and isinstance(knobs["reps"], int)
    assert knobs["qub_gain"] == 0.2
    assert knobs["qub_ch"] == 3

    # a sweep knob now accepts a SweepValue (the 160b typed sweep widget edits it)
    ctrl.set_node_params(
        index, {"detune_sweep": SweepValue(start=-30.0, stop=40.0, expts=71)}
    )
    detune = node.schema.lower(None)["detune_sweep"]
    assert (float(detune.start), float(detune.stop), int(detune.expts)) == (
        -30.0,
        40.0,
        71,
    )

    # an unknown key fast-fails (a real typo — the form only renders declared knobs)
    with pytest.raises(KeyError, match="Unknown node param"):
        ctrl.set_node_params(index, {"not_a_knob": 1})


# --- 3. seam invariant: only cfg/ imports gui.app.main from the package ---------


def test_only_cfg_seam_imports_measure_app():
    pkg = pathlib.Path(__file__).resolve().parents[2] / (
        "lib/zcu_tools/gui/app/autofluxdep"
    )
    allowed = {
        pkg / "cfg" / "__init__.py",
        pkg / "cfg" / "schema.py",
        pkg / "cfg" / "form.py",
    }
    offenders: list[str] = []
    for py in pkg.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom):
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = ";".join(a.name for a in node.names)
            if mod and "zcu_tools.gui.app.main" in mod and py not in allowed:
                offenders.append(f"{py.relative_to(pkg)}: {mod}")
    assert not offenders, "non-seam imports of gui.app.main:\n" + "\n".join(offenders)
