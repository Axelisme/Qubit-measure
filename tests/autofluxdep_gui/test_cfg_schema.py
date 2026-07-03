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
import pytest
from zcu_tools.gui.app.autofluxdep.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    FloatSpec,
    IntSpec,
    NodeCfgSchema,
    NodeFieldSpec,
    NodeSectionSpec,
    SweepSpec,
    SweepValue,
    node_field,
    node_section,
    sectioned_node_schema,
)
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder
from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

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


_BUILDERS: tuple[Builder, ...] = (
    QubitFreqBuilder(),
    LenRabiBuilder(),
    RoOptimizeBuilder(),
    T1Builder(),
    T2RamseyBuilder(),
    T2EchoBuilder(),
    MistBuilder(),
)

_BUILDER_IDS = [builder.name for builder in _BUILDERS]


def _sectioned_test_schema() -> NodeCfgSchema:
    return sectioned_node_schema(
        (
            node_section(
                "sweep",
                "Sweep",
                node_field(
                    "detune_sweep",
                    "detune",
                    SweepSpec(label="Detune"),
                    SweepValue(start=-20.0, stop=50.0, expts=141),
                ),
            ),
            node_section(
                "acquire",
                "Acquisition",
                node_field(
                    "reps",
                    "reps",
                    IntSpec("Reps"),
                    1000,
                ),
                node_field(
                    "earlystop_snr",
                    "earlystop_snr",
                    FloatSpec("Early-stop SNR", optional=True),
                    50.0,
                ),
            ),
            node_section(
                "drive",
                "Drive",
                node_field(
                    "qub_gain",
                    "gain",
                    FloatSpec("Gain"),
                    0.05,
                ),
            ),
        )
    )


def _assert_no_value_objects(value: object) -> None:
    assert not isinstance(value, (CfgSectionValue, DirectValue, EvalValue, SweepValue))
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_value_objects(child)


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
        "drive_gain_mode",
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
        "center_freq_mode",
        "center_freq",
        "center_gain_mode",
        "center_gain",
        "relax_delay_mode",
        "relax_delay",
        "skew_penalty",
    },
    "t1": {
        "sweep_range",
        "reps",
        "rounds",
        "earlystop_snr",
        "sweep_range_mode",
        "relax_delay_mode",
        "relax_delay",
    },
    "t2ramsey": {
        "sweep_range",
        "detune_ratio",
        "reps",
        "rounds",
        "earlystop_snr",
        "sweep_range_mode",
        "relax_delay_mode",
        "relax_delay",
    },
    "t2echo": {
        "sweep_range",
        "detune_ratio",
        "reps",
        "rounds",
        "earlystop_snr",
        "sweep_range_mode",
        "relax_delay_mode",
        "relax_delay",
    },
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

_EXPECTED_PATHS = {
    "qubit_freq": {
        "detune_sweep": "sweep.detune",
        "reps": "acquire.reps",
        "rounds": "acquire.rounds",
        "relax_delay": "acquire.relax_delay",
        "earlystop_snr": "acquire.earlystop_snr",
        "qub_waveform": "drive.waveform",
        "qub_ch": "drive.ch",
        "qub_nqz": "drive.nqz",
        "qub_gain": "drive.gain",
        "qub_length": "drive.length",
        "drive_gain_mode": "generation.drive_gain_mode",
    },
    "lenrabi": {
        "sweep_range": "sweep.length",
        "reps": "acquire.reps",
        "rounds": "acquire.rounds",
        "relax_delay": "acquire.relax_delay",
        "earlystop_snr": "acquire.earlystop_snr",
        "qub_waveform": "drive.waveform",
        "qub_ch": "drive.ch",
        "qub_nqz": "drive.nqz",
        "qub_gain": "drive.gain",
        "qub_length": "drive.length",
    },
    "ro_optimize": {
        "freq_expts": "grid.freq_points",
        "gain_expts": "grid.gain_points",
        "reps": "acquire.reps",
        "rounds": "acquire.rounds",
        "freq_window": "window.freq_half_width",
        "gain_window": "window.gain_half_width",
        "center_freq_mode": "generation.center_freq_mode",
        "center_freq": "generation.center_freq",
        "center_gain_mode": "generation.center_gain_mode",
        "center_gain": "generation.center_gain",
        "relax_delay_mode": "generation.relax_delay_mode",
        "relax_delay": "generation.relax_delay",
        "skew_penalty": "acquire.skew_penalty",
    },
    "t1": {
        "sweep_range": "sweep.delay",
        "reps": "acquire.reps",
        "rounds": "acquire.rounds",
        "earlystop_snr": "acquire.earlystop_snr",
        "sweep_range_mode": "generation.sweep_range_mode",
        "relax_delay_mode": "generation.relax_delay_mode",
        "relax_delay": "generation.relax_delay",
    },
    "t2ramsey": {
        "sweep_range": "sweep.delay",
        "detune_ratio": "ramsey.detune_ratio",
        "reps": "acquire.reps",
        "rounds": "acquire.rounds",
        "earlystop_snr": "acquire.earlystop_snr",
        "sweep_range_mode": "generation.sweep_range_mode",
        "relax_delay_mode": "generation.relax_delay_mode",
        "relax_delay": "generation.relax_delay",
    },
    "t2echo": {
        "sweep_range": "sweep.delay",
        "detune_ratio": "echo.detune_ratio",
        "reps": "acquire.reps",
        "rounds": "acquire.rounds",
        "earlystop_snr": "acquire.earlystop_snr",
        "sweep_range_mode": "generation.sweep_range_mode",
        "relax_delay_mode": "generation.relax_delay_mode",
        "relax_delay": "generation.relax_delay",
    },
    "mist": {
        "gain_sweep": "sweep.gain",
        "reps": "acquire.reps",
        "rounds": "acquire.rounds",
        "relax_delay": "acquire.relax_delay",
        "mist_waveform": "disturbance.waveform",
        "mist_ch": "disturbance.ch",
        "mist_nqz": "disturbance.nqz",
        "mist_freq": "disturbance.freq",
        "mist_gain": "disturbance.gain",
        "mist_length": "disturbance.length",
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
    "qfw_factor",
}


def test_expected_key_blocks_are_declared_user_knobs_only():
    """The exact key goldens describe declared knobs, not derived cfg fields."""

    builder_names = {builder.name for builder in _BUILDERS}
    assert set(_EXPECTED_KEYS) == builder_names
    assert set(_EXPECTED_PATHS) == builder_names
    for builder in _BUILDERS:
        keys = _EXPECTED_KEYS[builder.name]
        assert keys == set(_EXPECTED_PATHS[builder.name])
        assert keys.isdisjoint(_DERIVED_FORBIDDEN), builder.name


@pytest.mark.parametrize("builder", _BUILDERS, ids=_BUILDER_IDS)
def test_schema_keys_match_declared_knobs(builder: Builder):
    schema = builder.make_default_schema()
    assert set(schema.keys) == _EXPECTED_KEYS[builder.name], builder.name


@pytest.mark.parametrize("builder", _BUILDERS, ids=_BUILDER_IDS)
def test_schema_paths_match_section_layouts(builder: Builder):
    schema = builder.make_default_schema()
    assert {
        logical_key: schema.path_for(logical_key) for logical_key in schema.keys
    } == _EXPECTED_PATHS[builder.name], builder.name


@pytest.mark.parametrize("builder", _BUILDERS, ids=_BUILDER_IDS)
def test_no_derived_field_in_any_spec(builder: Builder):
    keys = set(builder.make_default_schema().keys)
    assert keys.isdisjoint(_DERIVED_FORBIDDEN), (builder.name, keys)


def test_sectioned_schema_lower_projects_logical_keys():
    schema = _sectioned_test_schema()

    assert schema.keys == ("detune_sweep", "reps", "earlystop_snr", "qub_gain")
    assert schema.path_for("detune_sweep") == "sweep.detune"
    assert schema.path_for("qub_gain") == "drive.gain"

    knobs = schema.lower(None)

    assert set(knobs) == {"detune_sweep", "reps", "earlystop_snr", "qub_gain"}
    assert "sweep" not in knobs
    assert "acquire" not in knobs
    detune = knobs["detune_sweep"]
    assert (float(detune.start), float(detune.stop), int(detune.expts)) == (
        -20.0,
        50.0,
        141,
    )
    assert knobs["reps"] == 1000
    assert knobs["earlystop_snr"] == 50.0
    assert knobs["qub_gain"] == 0.05

    schema.set_field("earlystop_snr", None)
    assert "earlystop_snr" not in schema.lower(None)


def test_sectioned_schema_set_field_and_with_overrides_write_nested_leaf():
    md = MetaDict()
    md.gain = 0.2
    schema = _sectioned_test_schema()

    schema.set_field("reps", "250")
    schema.with_overrides(
        {
            "detune_sweep": SweepValue(start=-5.0, stop=5.0, expts=11),
            "qub_gain": EvalValue("gain"),
        }
    )

    acquire = schema.schema.value.fields["acquire"]
    drive = schema.schema.value.fields["drive"]
    assert isinstance(acquire, CfgSectionValue)
    assert isinstance(drive, CfgSectionValue)
    reps = acquire.fields["reps"]
    assert isinstance(reps, DirectValue)
    assert reps.value == 250
    assert isinstance(drive.fields["gain"], EvalValue)

    knobs = schema.lower(None, md=md)
    detune = knobs["detune_sweep"]
    assert (float(detune.start), float(detune.stop), int(detune.expts)) == (
        -5.0,
        5.0,
        11,
    )
    assert knobs["qub_gain"] == 0.2


def test_sectioned_schema_read_knobs_is_flat_json_friendly():
    schema = _sectioned_test_schema()
    schema.set_field("qub_gain", EvalValue("gain"))
    schema.set_field(
        "detune_sweep",
        SweepValue(
            start=EvalValue("center - 2"),
            stop=EvalValue("center + 2"),
            expts=5,
        ),
    )

    knobs = schema.read_knobs()

    assert set(knobs) == {"detune_sweep", "reps", "earlystop_snr", "qub_gain"}
    assert "sweep" not in knobs
    assert "acquire" not in knobs
    assert knobs["detune_sweep"]["start"] == {
        "__kind": "eval",
        "expr": "center - 2",
    }
    assert knobs["detune_sweep"]["stop"] == {
        "__kind": "eval",
        "expr": "center + 2",
    }
    assert knobs["detune_sweep"]["expts"] == 5
    assert knobs["reps"] == 1000
    assert knobs["qub_gain"] == {"__kind": "eval", "expr": "gain"}
    _assert_no_value_objects(knobs)


def test_sectioned_schema_read_value_tree_is_nested_json_friendly():
    schema = _sectioned_test_schema()
    schema.set_field("qub_gain", EvalValue("gain"))
    schema.set_field(
        "detune_sweep",
        SweepValue(
            start=EvalValue("center - 2"),
            stop=EvalValue("center + 2"),
            expts=5,
        ),
    )

    knobs = schema.read_value_tree()

    assert set(knobs) == {"sweep", "acquire", "drive"}
    assert knobs["sweep"]["detune"]["start"] == {
        "__kind": "eval",
        "expr": "center - 2",
    }
    assert knobs["sweep"]["detune"]["stop"] == {
        "__kind": "eval",
        "expr": "center + 2",
    }
    assert knobs["sweep"]["detune"]["expts"] == 5
    assert knobs["acquire"]["reps"] == 1000
    assert knobs["drive"]["gain"] == {"__kind": "eval", "expr": "gain"}
    _assert_no_value_objects(knobs)


def test_sectioned_schema_unknown_logical_key_fast_fails():
    schema = _sectioned_test_schema()

    with pytest.raises(KeyError, match="Unknown node param"):
        schema.path_for("not_a_knob")
    with pytest.raises(KeyError, match="Unknown node param"):
        schema.set_field("not_a_knob", 1)
    with pytest.raises(KeyError, match="Unknown node param"):
        schema.with_overrides({"not_a_knob": 1})


def test_sectioned_node_schema_unsupported_spec_fast_fails():
    with pytest.raises(TypeError, match="Unsupported node field spec"):
        sectioned_node_schema(
            (
                node_section(
                    "bad",
                    "Bad",
                    node_field(
                        "nested",
                        "nested",
                        CfgSectionSpec(),
                        None,
                    ),
                ),
            )
        )


def test_sectioned_node_schema_section_mismatch_fast_fails():
    with pytest.raises(ValueError, match="declares section"):
        sectioned_node_schema(
            (
                NodeSectionSpec(
                    key="right",
                    label="Right",
                    fields=(
                        NodeFieldSpec(
                            logical_key="reps",
                            section_key="wrong",
                            field_key="reps",
                            spec=IntSpec("Reps"),
                            default=1000,
                        ),
                    ),
                ),
            )
        )


def test_sectioned_node_schema_duplicate_logical_key_fast_fails():
    with pytest.raises(ValueError, match="Duplicate node logical key"):
        sectioned_node_schema(
            (
                node_section(
                    "a",
                    "A",
                    node_field(
                        "reps",
                        "reps",
                        IntSpec("Reps"),
                        1000,
                    ),
                ),
                node_section(
                    "b",
                    "B",
                    node_field(
                        "reps",
                        "rounds",
                        IntSpec("Rounds"),
                        10,
                    ),
                ),
            )
        )


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
    assert knobs["drive_gain_mode"] == "adaptive"
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
    assert knobs["center_freq_mode"] == "previous_best"
    assert knobs["center_freq"] == 6000.0
    assert knobs["center_gain_mode"] == "previous_best"
    assert knobs["center_gain"] == 0.5
    assert knobs["relax_delay_mode"] == "auto_t1"
    assert knobs["relax_delay"] == 30.0
    assert knobs["skew_penalty"] == 0.0
    # the window half-widths are flat scalars, NOT a SweepSpec (decision)
    spec = RoOptimizeBuilder().make_default_schema().schema.spec
    window = spec.fields["window"]
    assert isinstance(window, CfgSectionSpec)
    assert not isinstance(window.fields["freq_half_width"], SweepSpec)
    assert not isinstance(window.fields["gain_half_width"], SweepSpec)


def test_t1_default_knobs():
    knobs = T1Builder().make_default_schema().lower(None)
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 10
    assert knobs["earlystop_snr"] == 20.0
    assert knobs["sweep_range_mode"] == "auto_t1"
    assert knobs["relax_delay_mode"] == "auto_t1"
    assert knobs["relax_delay"] == 30.0
    sweep = knobs["sweep_range"]
    assert (float(sweep.start), float(sweep.stop), int(sweep.expts)) == (0.5, 60.0, 101)


def test_t2_default_knobs():
    for builder in (T2RamseyBuilder(), T2EchoBuilder()):
        knobs = builder.make_default_schema().lower(None)
        assert knobs["reps"] == 1000
        assert knobs["rounds"] == 10
        assert knobs["detune_ratio"] == 0.05
        assert knobs["earlystop_snr"] == 20.0
        assert knobs["relax_delay_mode"] == "auto_t1"
        assert knobs["relax_delay"] == 30.0
        assert knobs["sweep_range_mode"] == (
            "auto_t2r" if builder.name == "t2ramsey" else "auto_t2e"
        )
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
        {"predict_freq": 5135.0, "qfw_factor": None}, modules={"readout": _READOUT}
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


def test_qubit_freq_make_cfg_uses_smoothed_qfw_factor_for_drive_gain():
    ml = _ml()
    builder = QubitFreqBuilder()
    env = RunEnv(
        flux=0.0,
        flux_idx=1,
        schema=builder.make_default_schema(),
        ml=ml,
    )

    cfg = builder.make_cfg(
        env,
        Snapshot(
            {"predict_freq": 5135.0, "qfw_factor": 65.0}, modules={"readout": _READOUT}
        ),
    )
    assert float(cfg.modules.qub_pulse.gain) == 0.1

    clamped = builder.make_cfg(
        env,
        Snapshot(
            {"predict_freq": 5135.0, "qfw_factor": 3.0}, modules={"readout": _READOUT}
        ),
    )
    assert float(clamped.modules.qub_pulse.gain) == 1.0


def test_qubit_freq_make_cfg_uses_const_waveform_when_named_waveform_missing():
    builder = QubitFreqBuilder()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=builder.make_default_schema(),
        ml=ModuleLibrary(),
    )
    snap = Snapshot(
        {"predict_freq": 5135.0, "qfw_factor": None}, modules={"readout": _READOUT}
    )

    cfg = builder.make_cfg(env, snap)

    assert cfg.modules.qub_pulse.waveform.style == "const"
    assert float(cfg.modules.qub_pulse.waveform.length) == 0.1


def test_lenrabi_make_cfg_uses_const_waveform_when_named_waveform_missing():
    builder = LenRabiBuilder()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=builder.make_default_schema(),
        ml=ModuleLibrary(),
    )
    snap = Snapshot({"qubit_freq": 5135.0}, modules={"opt_readout": _READOUT})

    cfg = builder.make_cfg(env, snap)

    assert cfg.modules.rabi_pulse.waveform.style == "const"
    assert float(cfg.modules.rabi_pulse.waveform.length) == 0.1


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


def test_mist_make_cfg_uses_const_waveform_when_named_waveform_missing():
    builder = MistBuilder()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=builder.make_default_schema(),
        ml=ModuleLibrary(),
    )
    snap = Snapshot(
        {"success": 1.0}, modules={"pi_pulse": _PI_PULSE, "opt_readout": _READOUT}
    )

    cfg = builder.make_cfg(env, snap)

    assert cfg.modules.mist_pulse.waveform.style == "const"
    assert float(cfg.modules.mist_pulse.waveform.length) == 0.1


# --- 2c. set_field type coercion + unknown-key fast-fail (the 160a bridge) ------


def test_set_field_coerces_text_to_type():
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("reps", "250")  # text from the prototype's line-edit form
    schema.set_field("qub_gain", "0.2")
    knobs = schema.lower(None)
    assert knobs["reps"] == 250 and isinstance(knobs["reps"], int)
    assert knobs["qub_gain"] == 0.2 and isinstance(knobs["qub_gain"], float)


def test_scalar_eval_value_lowers_against_md():
    md = MetaDict()
    md.gain = 0.125
    schema = QubitFreqBuilder().make_default_schema()

    schema.set_field("qub_gain", EvalValue("gain"))

    knobs = schema.lower(None, md=md)
    assert knobs["qub_gain"] == 0.125
    assert schema.read_knobs()["qub_gain"] == {
        "__kind": "eval",
        "expr": "gain",
    }
    assert schema.read_value_tree()["drive"]["gain"] == {
        "__kind": "eval",
        "expr": "gain",
    }


def test_sweep_eval_edges_lower_against_md():
    md = MetaDict()
    md.center = 10.0
    schema = QubitFreqBuilder().make_default_schema()

    schema.set_field(
        "detune_sweep",
        SweepValue(
            start=EvalValue("center - 2"),
            stop=EvalValue("center + 2"),
            expts=5,
        ),
    )

    detune = schema.lower(None, md=md)["detune_sweep"]
    assert (float(detune.start), float(detune.stop), int(detune.expts)) == (
        8.0,
        12.0,
        5,
    )
    assert schema.read_knobs()["detune_sweep"]["start"] == {
        "__kind": "eval",
        "expr": "center - 2",
    }
    assert schema.read_value_tree()["sweep"]["detune"]["start"] == {
        "__kind": "eval",
        "expr": "center - 2",
    }


def test_set_field_unknown_key_fast_fails():
    schema = QubitFreqBuilder().make_default_schema()
    with pytest.raises(KeyError, match="Unknown node param"):
        schema.set_field("not_a_knob", 1)


def test_with_overrides_unknown_key_fast_fails():
    schema = T1Builder().make_default_schema()
    with pytest.raises(KeyError, match="Unknown node param"):
        schema.with_overrides({"bogus": 1})


# --- 2d. set_node_params (controller typed entry): type, sweep, fast-fail -------


def test_set_node_params_types_and_fast_fails():
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
