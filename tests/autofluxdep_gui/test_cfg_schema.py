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
    node_path,
    node_section,
    path_node_schema,
    sectioned_node_schema,
    str_choice_spec,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgPersistenceError
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder
from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder
from zcu_tools.gui.app.autofluxdep.registry import create_placement
from zcu_tools.gui.session.types import ExpContext
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


def _ctx(md: MetaDict | None = None, ml: ModuleLibrary | None = None) -> ExpContext:
    return ExpContext(
        md=md if md is not None else MetaDict(),
        ml=ml if ml is not None else ModuleLibrary(),
        soc=None,
        soccfg=None,
    )


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
        "reset",
        "qub_pulse",
        "readout",
        "qub_ch",
        "qub_nqz",
        "qub_gain",
        "qub_length",
        "drive_gain_mode",
        "target_kappa",
        "max_drive_gain",
        "qf_width_seed",
        "qfw_seed_gain",
        "pred_freq_correction_enabled",
        "pred_freq_correction_strategy",
        "pred_freq_correction_idw_k",
        "pred_freq_correction_idw_epsilon",
    },
    "lenrabi": {
        "sweep_range",
        "reps",
        "rounds",
        "relax_delay",
        "earlystop_snr",
        "reset",
        "rabi_pulse",
        "readout",
        "qub_ch",
        "qub_nqz",
        "qub_gain",
        "relax_delay_mode",
        "t1_seed_us",
        "relax_factor",
        "relax_min_us",
        "sweep_range_mode",
        "expected_pi_length",
        "sweep_start_us",
        "sweep_stop_factor",
        "sweep_stop_min_us",
        "drive_gain_mode",
        "pi_product_seed",
        "pi_product_factor",
        "max_drive_gain",
        "pi_gain_feedback_enabled",
        "pi_gain_feedback_strategy",
        "pi_gain_feedback_step_gain",
    },
    "ro_optimize": {
        "freq_range",
        "gain_range",
        "reset",
        "pi_pulse",
        "readout",
        "reps",
        "rounds",
        "freq_range_mode",
        "gain_range_mode",
        "relax_delay_mode",
        "relax_delay",
        "skew_penalty",
        "t1_seed_us",
        "relax_factor",
        "freq_window_mode",
        "freq_half_width_mhz",
        "gain_window_mode",
        "gain_half_width",
    },
    "t1": {
        "sweep_range",
        "reset",
        "pi_pulse",
        "readout",
        "reps",
        "rounds",
        "earlystop_snr",
        "sweep_range_mode",
        "relax_delay_mode",
        "relax_delay",
        "t1_seed_us",
        "relax_factor",
        "relax_min_us",
        "sweep_start_us",
        "sweep_stop_factor",
        "sweep_stop_min_us",
    },
    "t2ramsey": {
        "sweep_range",
        "detune_ratio",
        "reset",
        "pi2_pulse",
        "readout",
        "reps",
        "rounds",
        "earlystop_snr",
        "sweep_range_mode",
        "relax_delay_mode",
        "relax_delay",
        "t1_seed_us",
        "t2r_seed_us",
        "relax_factor",
        "relax_min_us",
        "sweep_start_us",
        "sweep_stop_factor",
    },
    "t2echo": {
        "sweep_range",
        "detune_ratio",
        "reset",
        "pi_pulse",
        "pi2_pulse",
        "readout",
        "reps",
        "rounds",
        "earlystop_snr",
        "sweep_range_mode",
        "relax_delay_mode",
        "relax_delay",
        "t1_seed_us",
        "t2e_seed_us",
        "relax_factor",
        "relax_min_us",
        "sweep_start_us",
        "sweep_stop_factor",
        "fit_method",
    },
    "mist": {
        "gain_sweep",
        "reset",
        "pi_pulse",
        "mist_pulse",
        "readout",
        "reps",
        "rounds",
        "relax_delay",
        "mist_ch",
        "mist_nqz",
        "mist_freq",
        "mist_gain",
        "mist_length",
    },
}

_EXPECTED_PATHS = {
    "qubit_freq": {
        "detune_sweep": "sweep.freq",
        "reps": "reps",
        "rounds": "rounds",
        "relax_delay": "relax_delay",
        "earlystop_snr": "generation.safety.earlystop_snr",
        "reset": "modules.reset",
        "qub_pulse": "modules.qub_pulse",
        "readout": "modules.readout",
        "qub_ch": "modules.qub_pulse.ch",
        "qub_nqz": "modules.qub_pulse.nqz",
        "qub_gain": "modules.qub_pulse.gain",
        "qub_length": "modules.qub_pulse.waveform.length",
        "drive_gain_mode": "generation.feedback.drive_gain_mode",
        "target_kappa": "generation.feedback.target_kappa",
        "max_drive_gain": "generation.feedback.max_drive_gain",
        "qf_width_seed": "generation.feedback.qf_width_seed",
        "qfw_seed_gain": "generation.feedback.qfw_seed_gain",
        "pred_freq_correction_enabled": (
            "generation.feedback.pred_freq_correction_enabled"
        ),
        "pred_freq_correction_strategy": (
            "generation.feedback.pred_freq_correction_strategy"
        ),
        "pred_freq_correction_idw_k": (
            "generation.feedback.pred_freq_correction_idw_k"
        ),
        "pred_freq_correction_idw_epsilon": (
            "generation.feedback.pred_freq_correction_idw_epsilon"
        ),
    },
    "lenrabi": {
        "sweep_range": "sweep.length",
        "reps": "reps",
        "rounds": "rounds",
        "relax_delay": "relax_delay",
        "earlystop_snr": "generation.safety.earlystop_snr",
        "reset": "modules.reset",
        "rabi_pulse": "modules.qub_pulse",
        "readout": "modules.readout",
        "qub_ch": "modules.qub_pulse.ch",
        "qub_nqz": "modules.qub_pulse.nqz",
        "qub_gain": "modules.qub_pulse.gain",
        "relax_delay_mode": "generation.timing.relax_delay_mode",
        "t1_seed_us": "generation.timing.t1_seed_us",
        "relax_factor": "generation.timing.relax_factor",
        "relax_min_us": "generation.timing.relax_min_us",
        "sweep_range_mode": "generation.sweep.sweep_range_mode",
        "expected_pi_length": "generation.sweep.expected_pi_length",
        "sweep_start_us": "generation.sweep.sweep_start_us",
        "sweep_stop_factor": "generation.sweep.sweep_stop_factor",
        "sweep_stop_min_us": "generation.sweep.sweep_stop_min_us",
        "drive_gain_mode": "generation.feedback.drive_gain_mode",
        "pi_product_seed": "generation.feedback.pi_product_seed",
        "pi_product_factor": "generation.feedback.pi_product_factor",
        "max_drive_gain": "generation.feedback.max_drive_gain",
        "pi_gain_feedback_enabled": "generation.feedback.pi_gain_feedback_enabled",
        "pi_gain_feedback_strategy": "generation.feedback.pi_gain_feedback_strategy",
        "pi_gain_feedback_step_gain": (
            "generation.feedback.pi_gain_feedback_step_gain"
        ),
    },
    "ro_optimize": {
        "freq_range": "sweep.freq",
        "gain_range": "sweep.gain",
        "reset": "modules.reset",
        "pi_pulse": "modules.qub_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "relax_delay": "relax_delay",
        "freq_range_mode": "generation.sweep.freq_range_mode",
        "gain_range_mode": "generation.sweep.gain_range_mode",
        "relax_delay_mode": "generation.timing.relax_delay_mode",
        "skew_penalty": "skew_penalty",
        "t1_seed_us": "generation.timing.t1_seed_us",
        "relax_factor": "generation.timing.relax_factor",
        "freq_window_mode": "generation.feedback.freq_window_mode",
        "freq_half_width_mhz": "generation.feedback.freq_half_width_mhz",
        "gain_window_mode": "generation.feedback.gain_window_mode",
        "gain_half_width": "generation.feedback.gain_half_width",
    },
    "t1": {
        "sweep_range": "sweep.length",
        "reset": "modules.reset",
        "pi_pulse": "modules.pi_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "earlystop_snr": "generation.safety.earlystop_snr",
        "sweep_range_mode": "generation.sweep.sweep_range_mode",
        "relax_delay_mode": "generation.timing.relax_delay_mode",
        "relax_delay": "relax_delay",
        "t1_seed_us": "generation.timing.t1_seed_us",
        "relax_factor": "generation.timing.relax_factor",
        "relax_min_us": "generation.timing.relax_min_us",
        "sweep_start_us": "generation.sweep.sweep_start_us",
        "sweep_stop_factor": "generation.sweep.sweep_stop_factor",
        "sweep_stop_min_us": "generation.sweep.sweep_stop_min_us",
    },
    "t2ramsey": {
        "sweep_range": "sweep.length",
        "detune_ratio": "detune_ratio",
        "reset": "modules.reset",
        "pi2_pulse": "modules.pi2_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "earlystop_snr": "generation.safety.earlystop_snr",
        "sweep_range_mode": "generation.sweep.sweep_range_mode",
        "relax_delay_mode": "generation.timing.relax_delay_mode",
        "relax_delay": "relax_delay",
        "t1_seed_us": "generation.timing.t1_seed_us",
        "t2r_seed_us": "generation.timing.t2r_seed_us",
        "relax_factor": "generation.timing.relax_factor",
        "relax_min_us": "generation.timing.relax_min_us",
        "sweep_start_us": "generation.sweep.sweep_start_us",
        "sweep_stop_factor": "generation.sweep.sweep_stop_factor",
    },
    "t2echo": {
        "sweep_range": "sweep.length",
        "detune_ratio": "detune_ratio",
        "reset": "modules.reset",
        "pi_pulse": "modules.pi_pulse",
        "pi2_pulse": "modules.pi2_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "earlystop_snr": "generation.safety.earlystop_snr",
        "sweep_range_mode": "generation.sweep.sweep_range_mode",
        "relax_delay_mode": "generation.timing.relax_delay_mode",
        "relax_delay": "relax_delay",
        "t1_seed_us": "generation.timing.t1_seed_us",
        "t2e_seed_us": "generation.timing.t2e_seed_us",
        "relax_factor": "generation.timing.relax_factor",
        "relax_min_us": "generation.timing.relax_min_us",
        "sweep_start_us": "generation.sweep.sweep_start_us",
        "sweep_stop_factor": "generation.sweep.sweep_stop_factor",
        "fit_method": "generation.fit.fit_method",
    },
    "mist": {
        "gain_sweep": "sweep.gain",
        "reset": "modules.reset",
        "pi_pulse": "modules.init_pulse",
        "mist_pulse": "modules.probe_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "relax_delay": "relax_delay",
        "mist_ch": "modules.probe_pulse.ch",
        "mist_nqz": "modules.probe_pulse.nqz",
        "mist_freq": "modules.probe_pulse.freq",
        "mist_gain": "modules.probe_pulse.gain",
        "mist_length": "modules.probe_pulse.waveform.length",
    },
}

# The complete derived/upstream set that MUST NOT appear as a user knob (it is
# injected by make_cfg / produce from the predictor, prev-point fits, or modules).
_DERIVED_FORBIDDEN = {
    "predict_freq",
    "qubit_freq",
    "freq",
    "opt_readout",
    "best_ro_freq",
    "best_ro_gain",
    "t1",
    "t2r",
    "t2e",
    "fit_kappa",
    "qfw_factor",
    "pi_product",
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


def test_path_schema_renders_raw_cfg_tree_while_lowering_logical_keys():
    schema = path_node_schema(
        (
            node_path(
                "qub_gain",
                "modules.qub_pulse.gain",
                FloatSpec("gain"),
                0.05,
            ),
            node_path(
                "reps",
                "reps",
                IntSpec("reps"),
                1000,
            ),
            node_path(
                "drive_gain_mode",
                "generation.drive_gain_mode",
                str_choice_spec("drive_gain_mode", ("adaptive", "fixed")),
                "adaptive",
            ),
        ),
        section_labels={
            "modules": "modules",
            "modules.qub_pulse": "qub_pulse",
            "generation": "Generation overrides",
        },
    )

    value_tree = schema.read_value_tree()

    assert value_tree == {
        "modules": {
            "qub_pulse": {
                "gain": 0.05,
            },
        },
        "reps": 1000,
        "generation": {
            "drive_gain_mode": "adaptive",
        },
    }
    assert schema.path_for("qub_gain") == "modules.qub_pulse.gain"
    assert schema.lower(None) == {
        "qub_gain": 0.05,
        "reps": 1000,
        "drive_gain_mode": "adaptive",
    }


def test_generation_groups_keep_logical_knobs_flat():
    schema = QubitFreqBuilder().make_default_schema()

    generation_spec = schema.schema.spec.fields["generation"]
    generation_value = schema.schema.value.fields["generation"]
    assert isinstance(generation_spec, CfgSectionSpec)
    assert isinstance(generation_value, CfgSectionValue)
    assert set(generation_spec.fields) == {"feedback", "safety"}
    assert set(generation_value.fields) == {"feedback", "safety"}
    feedback_spec = generation_spec.fields["feedback"]
    feedback_value = generation_value.fields["feedback"]
    assert isinstance(feedback_spec, CfgSectionSpec)
    assert isinstance(feedback_value, CfgSectionValue)
    assert "drive_gain_mode" in feedback_spec.fields
    assert "drive_gain_mode" in feedback_value.fields
    assert schema.path_for("drive_gain_mode") == "generation.feedback.drive_gain_mode"

    knobs = schema.read_knobs()

    assert "drive_gain_mode" in knobs
    assert "feedback" not in knobs
    assert "safety" not in knobs
    assert schema.read_value_tree()["generation"]["feedback"]["drive_gain_mode"] == (
        "adaptive"
    )


def test_generation_persistence_uses_flat_logical_keys():
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("drive_gain_mode", "fixed")
    schema.set_field("earlystop_snr", 12.5)

    raw = schema.to_persisted_raw()

    generation = raw["generation"]
    assert isinstance(generation, dict)
    assert "feedback" not in generation
    assert "safety" not in generation
    assert generation["drive_gain_mode"] == {"__kind": "direct", "value": "fixed"}
    assert generation["earlystop_snr"] == {"__kind": "direct", "value": 12.5}

    restored = QubitFreqBuilder().make_default_schema()
    restored.restore_persisted_raw(raw)

    knobs = restored.read_knobs()
    assert knobs["drive_gain_mode"] == "fixed"
    assert knobs["earlystop_snr"] == pytest.approx(12.5)
    assert restored.read_value_tree()["generation"]["feedback"]["drive_gain_mode"] == (
        "fixed"
    )


def test_generation_restore_rejects_unknown_flat_persisted_key():
    schema = QubitFreqBuilder().make_default_schema()
    raw = schema.to_persisted_raw()
    generation = raw["generation"]
    assert isinstance(generation, dict)
    generation["not_a_generation_knob"] = {"__kind": "direct", "value": 1}

    with pytest.raises(NodeCfgPersistenceError, match="Unknown persisted generation"):
        QubitFreqBuilder().make_default_schema().restore_persisted_raw(raw)


def test_grouped_generation_section_is_removed_from_lower_raw():
    schema = path_node_schema(
        (
            node_path(
                "drive_gain_mode",
                "generation.feedback.drive_gain_mode",
                str_choice_spec("drive_gain_mode", ("adaptive", "fixed")),
                "adaptive",
            ),
            node_path("reps", "reps", IntSpec("reps"), 1000),
        ),
        section_labels={
            "generation": "Generation overrides",
            "generation.feedback": "Feedback / adaptive",
        },
    )

    raw = schema.lower_raw(None)

    assert raw == {"reps": 1000}


def test_path_schema_scalar_default_preserves_eval_value():
    md = MetaDict()
    md.gain = 0.25
    schema = path_node_schema(
        (
            node_path(
                "qub_gain",
                "modules.qub_pulse.gain",
                FloatSpec("gain"),
                EvalValue("gain"),
            ),
        ),
        section_labels={"modules": "modules", "modules.qub_pulse": "qub_pulse"},
    )

    assert schema.read_knobs()["qub_gain"] == {"__kind": "eval", "expr": "gain"}
    assert schema.read_value_tree()["modules"]["qub_pulse"]["gain"] == {
        "__kind": "eval",
        "expr": "gain",
    }
    assert schema.lower(None, md=md)["qub_gain"] == 0.25


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
    assert knobs["rounds"] == 10
    assert knobs["relax_delay"] == 1.0
    assert knobs["earlystop_snr"] == 50.0
    assert knobs["qub_ch"] == 0
    assert knobs["qub_nqz"] == 2
    assert knobs["qub_gain"] == 0.1
    assert knobs["qub_length"] == 5.0
    assert knobs["drive_gain_mode"] == "adaptive"
    assert knobs["target_kappa"] == 6.5
    assert knobs["max_drive_gain"] == 1.0
    assert knobs["qfw_seed_gain"] == 0.05
    # clearing optional defaults still omits them; channel/nqz remain required raw
    # cfg fields because PulseCfg cannot run without concrete hardware routing.
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("earlystop_snr", None)
    cleared = schema.lower(None)
    assert "earlystop_snr" not in cleared
    schema.set_field("qub_ch", None)
    with pytest.raises(RuntimeError, match="modules\\.qub_pulse\\.ch"):
        schema.lower(None)
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("qub_nqz", 0)
    with pytest.raises(RuntimeError, match="modules\\.qub_pulse\\.nqz.*choices"):
        schema.lower(None)
    # autofluxdep copies the adapter-shaped cfg, then replaces the absolute
    # twotone/freq sweep with md-style relative detune.
    detune = knobs["detune_sweep"]
    assert (float(detune.start), float(detune.stop), int(detune.expts)) == (
        -20.0,
        50.0,
        141,
    )


def test_lenrabi_default_knobs():
    knobs = LenRabiBuilder().make_default_schema().lower(None)
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 10
    assert knobs["relax_delay"] == 30.0
    assert knobs["earlystop_snr"] == 30.0
    assert knobs["relax_delay_mode"] == "auto_t1"
    assert knobs["t1_seed_us"] == 10.0
    assert knobs["relax_factor"] == 3.0
    assert knobs["relax_min_us"] == 0.0
    assert knobs["sweep_range_mode"] == "auto_pi_length"
    assert knobs["expected_pi_length"] == 1.0
    assert knobs["sweep_start_us"] == 0.05
    assert knobs["sweep_stop_factor"] == 5.0
    assert knobs["sweep_stop_min_us"] == 0.5
    assert knobs["drive_gain_mode"] == "auto_pi_product"
    assert knobs["pi_product_seed"] == 1.0
    assert knobs["pi_product_factor"] == 1.5
    assert knobs["max_drive_gain"] == 1.0
    assert knobs["qub_ch"] == 0
    assert knobs["qub_nqz"] == 2
    assert knobs["qub_gain"] == 0.3
    sweep = knobs["sweep_range"]
    assert np.allclose([float(sweep.start), float(sweep.stop)], [0.05, 5.0])
    assert int(sweep.expts) == 101


def test_mist_default_knobs():
    knobs = MistBuilder().make_default_schema().lower(None)
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 10
    assert knobs["relax_delay"] == 30.5
    assert knobs["mist_ch"] == 0
    assert knobs["mist_nqz"] == 2
    assert knobs["mist_freq"] == 6000.0
    assert knobs["mist_gain"] == 0.05
    assert knobs["mist_length"] == 1.0
    gain = knobs["gain_sweep"]
    assert (float(gain.start), float(gain.stop), int(gain.expts)) == (0.0, 1.0, 151)

    schema = MistBuilder().make_default_schema()
    schema.set_field("mist_nqz", 0)
    with pytest.raises(RuntimeError, match="modules\\.probe_pulse\\.nqz.*choices"):
        schema.lower(None)


def test_ro_optimize_default_knobs():
    knobs = RoOptimizeBuilder().make_default_schema().lower(None)
    freq_range = knobs["freq_range"]
    gain_range = knobs["gain_range"]
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 10
    assert (
        float(freq_range.start),
        float(freq_range.stop),
        int(freq_range.expts),
    ) == (5999.0, 6001.0, 31)
    assert np.allclose([float(gain_range.start), float(gain_range.stop)], [0.45, 0.55])
    assert int(gain_range.expts) == 31
    assert knobs["freq_range_mode"] == "previous_best"
    assert knobs["gain_range_mode"] == "previous_best"
    assert knobs["relax_delay_mode"] == "auto_t1"
    assert knobs["relax_delay"] == 30.0
    assert knobs["t1_seed_us"] == 10.0
    assert knobs["relax_factor"] == 3.0
    assert knobs["freq_window_mode"] == "fixed_half_width"
    assert knobs["freq_half_width_mhz"] == 1.0
    assert knobs["gain_window_mode"] == "fixed_half_width"
    assert knobs["gain_half_width"] == 0.05
    assert knobs["skew_penalty"] == 0.0
    # The sweep range widgets carry the user-facing start/stop/expts raw-cfg shape.
    spec = RoOptimizeBuilder().make_default_schema().schema.spec
    sweep = spec.fields["sweep"]
    assert isinstance(sweep, CfgSectionSpec)
    assert isinstance(sweep.fields["freq"], SweepSpec)
    assert isinstance(sweep.fields["gain"], SweepSpec)


def test_t1_default_knobs():
    knobs = T1Builder().make_default_schema().lower(None)
    assert knobs["reps"] == 1000
    assert knobs["rounds"] == 10
    assert knobs["earlystop_snr"] == 20.0
    assert knobs["sweep_range_mode"] == "auto_t1"
    assert knobs["relax_delay_mode"] == "auto_t1"
    assert knobs["relax_delay"] == 30.0
    assert knobs["t1_seed_us"] == 10.0
    assert knobs["relax_factor"] == 3.0
    assert knobs["relax_min_us"] == 1.0
    assert knobs["sweep_start_us"] == 0.5
    assert knobs["sweep_stop_factor"] == 5.0
    assert knobs["sweep_stop_min_us"] == 1.0
    sweep = knobs["sweep_range"]
    assert (float(sweep.start), float(sweep.stop), int(sweep.expts)) == (
        0.5,
        50.0,
        101,
    )


def test_t2_default_knobs():
    for builder in (T2RamseyBuilder(), T2EchoBuilder()):
        knobs = builder.make_default_schema().lower(None)
        assert knobs["reps"] == 1000
        assert knobs["rounds"] == 10
        assert knobs["detune_ratio"] == 0.05
        assert knobs["earlystop_snr"] == 20.0
        assert knobs["relax_delay_mode"] == "auto_t1"
        assert knobs["relax_delay"] == 30.0
        assert knobs["t1_seed_us"] == 10.0
        assert knobs["relax_factor"] == 3.0
        assert knobs["relax_min_us"] == 1.0
        assert knobs["sweep_start_us"] == 0.0
        assert knobs["sweep_stop_factor"] == 2.5
        assert knobs["sweep_range_mode"] == (
            "auto_t2r" if builder.name == "t2ramsey" else "auto_t2e"
        )
        if builder.name == "t2ramsey":
            assert knobs["t2r_seed_us"] == 5.0
        else:
            assert knobs["t2e_seed_us"] == 5.0
            assert knobs["fit_method"] == "auto_by_detune"
        sweep = knobs["sweep_range"]
        assert (float(sweep.start), float(sweep.stop), int(sweep.expts)) == (
            0.0,
            12.5,
            101,
        ), builder.name


def test_fresh_node_defaults_seed_from_md_values():
    md = MetaDict()
    md.qub_ch = 7
    md.t1 = 12.0
    md.pi_len = 0.2
    md.r_f = 6200.0
    md.rf_w = 10.0
    md.q_f = 5100.0
    md.qf_w = 20.0
    md.t2r = 8.0
    md.t2e = 9.0
    ctx = _ctx(md=md)

    qubit = create_placement("qubit_freq", ctx=ctx).schema.lower(None, md=md)
    assert qubit["qub_ch"] == 7

    lenrabi = LenRabiBuilder().make_default_schema(ctx).lower(None, md=md)
    assert lenrabi["qub_ch"] == 7
    assert lenrabi["t1_seed_us"] == 12.0
    assert lenrabi["expected_pi_length"] == 0.2
    assert lenrabi["relax_delay"] == 36.0
    assert np.allclose(
        [float(lenrabi["sweep_range"].start), float(lenrabi["sweep_range"].stop)],
        [0.05, 1.0],
    )

    ro = RoOptimizeBuilder().make_default_schema(ctx).lower(None, md=md)
    assert ro["t1_seed_us"] == 12.0
    assert ro["relax_delay"] == 36.0
    assert (
        float(ro["freq_range"].start),
        float(ro["freq_range"].stop),
        int(ro["freq_range"].expts),
    ) == (6199.0, 6201.0, 31)

    t1 = T1Builder().make_default_schema(ctx).lower(None, md=md)
    assert t1["t1_seed_us"] == 12.0
    assert t1["relax_delay"] == 36.0
    assert (float(t1["sweep_range"].start), float(t1["sweep_range"].stop)) == (
        0.5,
        60.0,
    )

    ramsey = T2RamseyBuilder().make_default_schema(ctx).lower(None, md=md)
    echo = T2EchoBuilder().make_default_schema(ctx).lower(None, md=md)
    assert ramsey["t1_seed_us"] == 12.0
    assert ramsey["t2r_seed_us"] == 8.0
    assert ramsey["relax_delay"] == 36.0
    assert (float(ramsey["sweep_range"].start), float(ramsey["sweep_range"].stop)) == (
        0.0,
        20.0,
    )
    assert echo["t1_seed_us"] == 12.0
    assert echo["t2e_seed_us"] == 9.0
    assert echo["relax_delay"] == 36.0
    assert (float(echo["sweep_range"].start), float(echo["sweep_range"].stop)) == (
        0.0,
        22.5,
    )

    mist = MistBuilder().make_default_schema(ctx).lower(None, md=md)
    assert mist["mist_ch"] == 0
    assert mist["mist_freq"] == 6200.0
    assert mist["relax_delay"] == 60.0


def test_fresh_node_defaults_seed_from_ml_modules():
    ml = ModuleLibrary()
    ml.register_waveform(
        qub_flat={
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "cosine", "length": 0.02},
        }
    )
    ml.register_module(
        pi_amp={
            "type": "pulse",
            "ch": 5,
            "nqz": 1,
            "freq": 5100.0,
            "gain": 0.6,
            "waveform": {"style": "const", "length": 0.24},
        },
        readout_dpm={
            "type": "readout/pulse",
            "pulse_cfg": {
                "type": "pulse",
                "ch": 2,
                "nqz": 2,
                "freq": 6201.0,
                "gain": 0.42,
                "waveform": {"style": "const", "length": 1.0},
            },
            "ro_cfg": {
                "type": "readout/direct",
                "ro_ch": 0,
                "ro_length": 0.9,
                "ro_freq": 6201.0,
                "trig_offset": 0.6,
            },
        },
    )
    ctx = _ctx(ml=ml)

    lenrabi = LenRabiBuilder().make_default_schema(ctx).lower(ml, md=ctx.md)
    assert lenrabi["qub_ch"] == 0
    assert lenrabi["qub_nqz"] == 2
    assert lenrabi["qub_gain"] == 0.3
    assert lenrabi["expected_pi_length"] == 0.24
    assert lenrabi["pi_product_seed"] == pytest.approx(0.144)
    assert np.allclose(
        [float(lenrabi["sweep_range"].start), float(lenrabi["sweep_range"].stop)],
        [0.05, 1.2],
    )

    ro = RoOptimizeBuilder().make_default_schema(ctx).lower(ml, md=ctx.md)
    assert np.allclose(
        [float(ro["freq_range"].start), float(ro["freq_range"].stop)],
        [6200.0, 6202.0],
    )
    assert np.allclose(
        [float(ro["gain_range"].start), float(ro["gain_range"].stop)],
        [0.37, 0.47],
    )


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
    # the builder defaults, now sourced from the schema
    assert cfg.reps == 1000
    assert cfg.rounds == 10
    assert cfg.relax_delay == 1.0
    assert int(cfg.modules.qub_pulse.ch) == 0
    assert int(cfg.modules.qub_pulse.nqz) == 2
    assert float(cfg.modules.qub_pulse.gain) == 0.1
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
    assert float(cfg.modules.qub_pulse.waveform.length) == 5.0


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
    assert float(cfg.modules.rabi_pulse.waveform.length) == 1.0


def test_mist_make_cfg_uses_schema_defaults():
    ml = ModuleLibrary()
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
    assert cfg.rounds == 10
    assert cfg.relax_delay == 30.5
    assert int(cfg.modules.mist_pulse.ch) == 0
    assert float(cfg.modules.mist_pulse.gain) == 0.05
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
    assert float(cfg.modules.mist_pulse.waveform.length) == 1.0


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
    assert schema.read_value_tree()["modules"]["qub_pulse"]["value"]["gain"] == {
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
    assert schema.read_value_tree()["sweep"]["freq"]["start"] == {
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
