"""Typed node-knob CfgSchema: structure, defaults, equivalence, seam.

Three families of test:

1. **Structure** — each node's ``make_default_schema`` declares exactly the user
   knobs (the typed node settings), and *no* derived/upstream field (predict_freq,
   relax=3·T1, the pi/readout modules) leaks into the spec.
2. **Defaults** — default schemas lower through the same schema/helper paths as
   production. Tests assert invariants and derive expected values from production
   schemas/helpers instead of duplicating default tables.
3. **Seam invariant** — only the ``cfg/`` seam (``__init__.py`` / ``schema.py`` /
   ``form.py``) may import ``zcu_tools.gui.app.main`` from inside the autofluxdep
   package.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any, cast

import pytest
from zcu_tools.gui.app.autofluxdep.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    FloatSpec,
    IntSpec,
    ModuleRefSpec,
    NodeCfgSchema,
    OverridePath,
    OverridePlan,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    apply_override_patches,
    override_plan_to_wire,
    str_choice_spec,
    validate_override_plan_base_cfg,
)
from zcu_tools.gui.app.autofluxdep.cfg.schema import NodeCfgPersistenceError
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.defaults import pulse_length, pulse_product
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.app.autofluxdep.nodes.lenrabi import LenRabiBuilder
from zcu_tools.gui.app.autofluxdep.nodes.mist import MistBuilder
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.app.autofluxdep.nodes.readout_defaults import (
    seed_readout_freq,
    seed_readout_gain,
)
from zcu_tools.gui.app.autofluxdep.nodes.ro_optimize import RoOptimizeBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t1 import T1Builder
from zcu_tools.gui.app.autofluxdep.nodes.t2echo import T2EchoBuilder
from zcu_tools.gui.app.autofluxdep.nodes.t2ramsey import T2RamseyBuilder
from zcu_tools.gui.app.autofluxdep.nodes.timing_defaults import (
    auto_relax_delay_from_t1,
    auto_stop_sweep_range,
)
from zcu_tools.gui.app.autofluxdep.registry import create_placement
from zcu_tools.gui.session.types import ExpContext
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import PulseReadoutCfg

from ._helpers import (
    NodeFieldSpec,
    NodeSectionSpec,
    node_field,
    node_path,
    node_section,
    path_node_schema,
    read_value_tree,
    sectioned_node_schema,
)

_READOUT = {
    "type": "readout/pulse",
    "pulse_cfg": {
        "ch": 0,
        "nqz": 2,
        "freq": 7444.6,
        "gain": 1.0,
        "waveform": {"style": "const", "length": 1.0},
    },
    "ro_cfg": {"ro_ch": 0, "ro_freq": 7444.6, "ro_length": 0.9, "trig_offset": 0.6},
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


def _section_spec(schema: NodeCfgSchema, key: str) -> CfgSectionSpec:
    section = schema.schema.spec.fields[key]
    assert isinstance(section, CfgSectionSpec)
    return section


def _generation_group_spec(schema: NodeCfgSchema, key: str) -> CfgSectionSpec:
    group = _section_spec(schema, "generation").fields[key]
    assert isinstance(group, CfgSectionSpec)
    return group


def _scalar_labels(section: CfgSectionSpec) -> dict[str, str]:
    labels: dict[str, str] = {}
    for key, spec in section.fields.items():
        assert isinstance(spec, ScalarSpec)
        labels[key] = spec.label
    return labels


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


def _assert_sweep_bounds(sweep: Any, expected: tuple[float, float]) -> None:
    assert (float(sweep.start), float(sweep.stop)) == pytest.approx(expected)


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
        "acquire_retry",
        "qub_pulse",
        "readout",
        "qub_ch",
        "qub_nqz",
        "qub_gain",
        "qub_length",
        "drive_gain_mode",
        "target_kappa",
        "qf_width_seed",
        "physical_recovery_mode",
        "pred_freq_correction_strategy",
        "pred_freq_correction_idw_k",
        "pred_freq_correction_idw_epsilon",
        "pred_freq_correction_decay_points",
    },
    "lenrabi": {
        "sweep_range",
        "reps",
        "rounds",
        "relax_delay",
        "earlystop_snr",
        "acquire_retry",
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
        "pi_gain_feedback_strategy",
    },
    "ro_optimize": {
        "freq_range",
        "gain_range",
        "pi_pulse",
        "readout",
        "reps",
        "rounds",
        "acquire_retry",
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
        "pi_pulse",
        "readout",
        "reps",
        "rounds",
        "earlystop_snr",
        "acquire_retry",
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
        "pi2_pulse",
        "readout",
        "reps",
        "rounds",
        "earlystop_snr",
        "acquire_retry",
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
        "pi_pulse",
        "pi2_pulse",
        "readout",
        "reps",
        "rounds",
        "earlystop_snr",
        "acquire_retry",
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
        "pi_pulse",
        "mist_pulse",
        "readout",
        "reps",
        "rounds",
        "relax_delay",
        "acquire_retry",
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
        "earlystop_snr": "generation.acquisition.earlystop_snr",
        "acquire_retry": "generation.acquisition.acquire_retry",
        "qub_pulse": "modules.qub_pulse",
        "readout": "modules.readout",
        "qub_ch": "modules.qub_pulse.ch",
        "qub_nqz": "modules.qub_pulse.nqz",
        "qub_gain": "modules.qub_pulse.gain",
        "qub_length": "modules.qub_pulse.waveform.length",
        "drive_gain_mode": "generation.drive_gain.drive_gain_mode",
        "target_kappa": "generation.drive_gain.target_kappa",
        "qf_width_seed": "generation.drive_gain.qf_width_seed",
        "physical_recovery_mode": "generation.freq_recovery.physical_recovery_mode",
        "pred_freq_correction_strategy": (
            "generation.predictor_correction.pred_freq_correction_strategy"
        ),
        "pred_freq_correction_idw_k": (
            "generation.predictor_correction.pred_freq_correction_idw_k"
        ),
        "pred_freq_correction_idw_epsilon": (
            "generation.predictor_correction.pred_freq_correction_idw_epsilon"
        ),
        "pred_freq_correction_decay_points": (
            "generation.predictor_correction.pred_freq_correction_decay_points"
        ),
    },
    "lenrabi": {
        "sweep_range": "sweep.length",
        "reps": "reps",
        "rounds": "rounds",
        "relax_delay": "relax_delay",
        "earlystop_snr": "generation.acquisition.earlystop_snr",
        "acquire_retry": "generation.acquisition.acquire_retry",
        "rabi_pulse": "modules.rabi_pulse",
        "readout": "modules.readout",
        "qub_ch": "modules.rabi_pulse.ch",
        "qub_nqz": "modules.rabi_pulse.nqz",
        "qub_gain": "modules.rabi_pulse.gain",
        "relax_delay_mode": "generation.relax.relax_delay_mode",
        "t1_seed_us": "generation.relax.t1_seed_us",
        "relax_factor": "generation.relax.relax_factor",
        "relax_min_us": "generation.relax.relax_min_us",
        "sweep_range_mode": "generation.sweep.sweep_range_mode",
        "expected_pi_length": "generation.sweep.expected_pi_length",
        "sweep_start_us": "generation.sweep.sweep_start_us",
        "sweep_stop_factor": "generation.sweep.sweep_stop_factor",
        "sweep_stop_min_us": "generation.sweep.sweep_stop_min_us",
        "drive_gain_mode": "generation.drive_gain.drive_gain_mode",
        "pi_product_seed": "generation.drive_gain.pi_product_seed",
        "pi_gain_feedback_strategy": "generation.pi_feedback.pi_gain_feedback_strategy",
    },
    "ro_optimize": {
        "freq_range": "sweep.freq",
        "gain_range": "sweep.gain",
        "pi_pulse": "modules.pi_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "relax_delay": "relax_delay",
        "acquire_retry": "generation.acquisition.acquire_retry",
        "freq_range_mode": "generation.freq_search.freq_range_mode",
        "gain_range_mode": "generation.gain_search.gain_range_mode",
        "relax_delay_mode": "generation.relax.relax_delay_mode",
        "skew_penalty": "skew_penalty",
        "t1_seed_us": "generation.relax.t1_seed_us",
        "relax_factor": "generation.relax.relax_factor",
        "freq_window_mode": "generation.freq_search.freq_window_mode",
        "freq_half_width_mhz": "generation.freq_search.freq_half_width_mhz",
        "gain_window_mode": "generation.gain_search.gain_window_mode",
        "gain_half_width": "generation.gain_search.gain_half_width",
    },
    "t1": {
        "sweep_range": "sweep.length",
        "pi_pulse": "modules.pi_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "earlystop_snr": "generation.acquisition.earlystop_snr",
        "acquire_retry": "generation.acquisition.acquire_retry",
        "sweep_range_mode": "generation.sweep.sweep_range_mode",
        "relax_delay_mode": "generation.relax.relax_delay_mode",
        "relax_delay": "relax_delay",
        "t1_seed_us": "generation.relax.t1_seed_us",
        "relax_factor": "generation.relax.relax_factor",
        "relax_min_us": "generation.relax.relax_min_us",
        "sweep_start_us": "generation.sweep.sweep_start_us",
        "sweep_stop_factor": "generation.sweep.sweep_stop_factor",
        "sweep_stop_min_us": "generation.sweep.sweep_stop_min_us",
    },
    "t2ramsey": {
        "sweep_range": "sweep.length",
        "detune_ratio": "detune_ratio",
        "pi2_pulse": "modules.pi2_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "earlystop_snr": "generation.acquisition.earlystop_snr",
        "acquire_retry": "generation.acquisition.acquire_retry",
        "sweep_range_mode": "generation.sweep.sweep_range_mode",
        "relax_delay_mode": "generation.relax.relax_delay_mode",
        "relax_delay": "relax_delay",
        "t1_seed_us": "generation.relax.t1_seed_us",
        "t2r_seed_us": "generation.sweep.t2r_seed_us",
        "relax_factor": "generation.relax.relax_factor",
        "relax_min_us": "generation.relax.relax_min_us",
        "sweep_start_us": "generation.sweep.sweep_start_us",
        "sweep_stop_factor": "generation.sweep.sweep_stop_factor",
    },
    "t2echo": {
        "sweep_range": "sweep.length",
        "detune_ratio": "detune_ratio",
        "pi_pulse": "modules.pi_pulse",
        "pi2_pulse": "modules.pi2_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "earlystop_snr": "generation.acquisition.earlystop_snr",
        "acquire_retry": "generation.acquisition.acquire_retry",
        "sweep_range_mode": "generation.sweep.sweep_range_mode",
        "relax_delay_mode": "generation.relax.relax_delay_mode",
        "relax_delay": "relax_delay",
        "t1_seed_us": "generation.relax.t1_seed_us",
        "t2e_seed_us": "generation.sweep.t2e_seed_us",
        "relax_factor": "generation.relax.relax_factor",
        "relax_min_us": "generation.relax.relax_min_us",
        "sweep_start_us": "generation.sweep.sweep_start_us",
        "sweep_stop_factor": "generation.sweep.sweep_stop_factor",
        "fit_method": "generation.fit.fit_method",
    },
    "mist": {
        "gain_sweep": "sweep.gain",
        "pi_pulse": "modules.pi_pulse",
        "mist_pulse": "modules.mist_pulse",
        "readout": "modules.readout",
        "reps": "reps",
        "rounds": "rounds",
        "relax_delay": "relax_delay",
        "acquire_retry": "generation.acquisition.acquire_retry",
        "mist_ch": "modules.mist_pulse.ch",
        "mist_nqz": "modules.mist_pulse.nqz",
        "mist_freq": "modules.mist_pulse.freq",
        "mist_gain": "modules.mist_pulse.gain",
        "mist_length": "modules.mist_pulse.waveform.length",
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
    assert dict(schema.logical_paths) == _EXPECTED_PATHS[builder.name], builder.name


@pytest.mark.parametrize("builder", _BUILDERS, ids=_BUILDER_IDS)
def test_no_derived_field_in_any_spec(builder: Builder):
    keys = set(builder.make_default_schema().keys)
    assert keys.isdisjoint(_DERIVED_FORBIDDEN), (builder.name, keys)


def test_sectioned_schema_lower_projects_logical_keys():
    schema = _sectioned_test_schema()

    assert schema.keys == ("detune_sweep", "reps", "earlystop_snr", "qub_gain")
    assert schema.logical_paths["detune_sweep"] == "sweep.detune"
    assert schema.logical_paths["qub_gain"] == "drive.gain"

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

    value_tree = read_value_tree(schema)

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
    assert schema.logical_paths["qub_gain"] == "modules.qub_pulse.gain"
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
    assert set(generation_spec.fields) == {
        "acquisition",
        "drive_gain",
        "freq_recovery",
        "predictor_correction",
    }
    assert set(generation_value.fields) == {
        "acquisition",
        "drive_gain",
        "freq_recovery",
        "predictor_correction",
    }
    acquisition_spec = generation_spec.fields["acquisition"]
    acquisition_value = generation_value.fields["acquisition"]
    assert isinstance(acquisition_spec, CfgSectionSpec)
    assert isinstance(acquisition_value, CfgSectionValue)
    assert set(acquisition_spec.fields) == {"earlystop_snr", "acquire_retry"}
    assert set(acquisition_value.fields) == {"earlystop_snr", "acquire_retry"}
    drive_gain_spec = generation_spec.fields["drive_gain"]
    drive_gain_value = generation_value.fields["drive_gain"]
    assert isinstance(drive_gain_spec, CfgSectionSpec)
    assert isinstance(drive_gain_value, CfgSectionValue)
    assert "drive_gain_mode" in drive_gain_spec.fields
    assert "drive_gain_mode" in drive_gain_value.fields
    recovery_spec = generation_spec.fields["freq_recovery"]
    recovery_value = generation_value.fields["freq_recovery"]
    assert isinstance(recovery_spec, CfgSectionSpec)
    assert isinstance(recovery_value, CfgSectionValue)
    assert "physical_recovery_mode" in recovery_spec.fields
    assert "physical_recovery_mode" in recovery_value.fields
    assert (
        schema.logical_paths["drive_gain_mode"]
        == "generation.drive_gain.drive_gain_mode"
    )
    assert (
        schema.logical_paths["physical_recovery_mode"]
        == "generation.freq_recovery.physical_recovery_mode"
    )

    knobs = schema.read_knobs()

    assert "drive_gain_mode" in knobs
    assert "physical_recovery_mode" in knobs
    assert knobs["acquire_retry"] == 3
    assert "drive_gain" not in knobs
    assert "freq_recovery" not in knobs
    assert "acquisition" not in knobs
    assert read_value_tree(schema)["generation"]["drive_gain"]["drive_gain_mode"] == (
        "adaptive"
    )


def test_generation_display_labels_drop_redundant_group_prefixes():
    qf_schema = QubitFreqBuilder().make_default_schema()
    assert _scalar_labels(_generation_group_spec(qf_schema, "freq_recovery")) == {
        "physical_recovery_mode": "mode",
    }
    assert _scalar_labels(_generation_group_spec(qf_schema, "drive_gain")) == {
        "drive_gain_mode": "mode",
        "target_kappa": "target_kappa",
        "qf_width_seed": "initial_linewidth_mhz",
    }
    assert _scalar_labels(
        _generation_group_spec(qf_schema, "predictor_correction")
    ) == {
        "pred_freq_correction_strategy": "strategy",
        "pred_freq_correction_idw_k": "idw_k",
        "pred_freq_correction_idw_epsilon": "idw_epsilon",
        "pred_freq_correction_decay_points": "decay_points",
    }
    pred_strategy = _generation_group_spec(qf_schema, "predictor_correction").fields[
        "pred_freq_correction_strategy"
    ]
    assert isinstance(pred_strategy, ScalarSpec)
    assert pred_strategy.choices == ["off", "idw", "last_good"]

    lenrabi_schema = LenRabiBuilder().make_default_schema()
    assert _scalar_labels(_generation_group_spec(lenrabi_schema, "relax")) == {
        "relax_delay_mode": "delay_mode",
        "t1_seed_us": "initial_t1_us",
        "relax_factor": "factor",
        "relax_min_us": "min_us",
    }
    assert _scalar_labels(_generation_group_spec(lenrabi_schema, "sweep")) == {
        "sweep_range_mode": "range_mode",
        "expected_pi_length": "target_pi_length_us",
        "sweep_start_us": "start_us",
        "sweep_stop_factor": "stop_factor",
        "sweep_stop_min_us": "stop_min_us",
    }
    assert _scalar_labels(_generation_group_spec(lenrabi_schema, "drive_gain")) == {
        "drive_gain_mode": "mode",
        "pi_product_seed": "initial_pi_product",
    }
    assert _scalar_labels(_generation_group_spec(lenrabi_schema, "pi_feedback")) == {
        "pi_gain_feedback_strategy": "strategy",
    }
    pi_strategy = _generation_group_spec(lenrabi_schema, "pi_feedback").fields[
        "pi_gain_feedback_strategy"
    ]
    assert isinstance(pi_strategy, ScalarSpec)
    assert pi_strategy.choices == ["off", "log_step"]

    ramsey_schema = T2RamseyBuilder().make_default_schema()
    assert _scalar_labels(_generation_group_spec(ramsey_schema, "sweep")) == {
        "sweep_range_mode": "range_mode",
        "t2r_seed_us": "initial_t2r_us",
        "sweep_start_us": "start_us",
        "sweep_stop_factor": "stop_factor",
    }

    ro_schema = RoOptimizeBuilder().make_default_schema()
    assert _scalar_labels(_generation_group_spec(ro_schema, "freq_search")) == {
        "freq_range_mode": "freq_mode",
        "freq_window_mode": "freq_mode",
        "freq_half_width_mhz": "freq_half_width_mhz",
    }
    assert _scalar_labels(_generation_group_spec(ro_schema, "gain_search")) == {
        "gain_range_mode": "gain_mode",
        "gain_window_mode": "gain_mode",
        "gain_half_width": "gain_half_width",
    }

    echo_schema = T2EchoBuilder().make_default_schema()
    assert _scalar_labels(_generation_group_spec(echo_schema, "sweep")) == {
        "sweep_range_mode": "range_mode",
        "t2e_seed_us": "initial_t2e_us",
        "sweep_start_us": "start_us",
        "sweep_stop_factor": "stop_factor",
    }
    assert _scalar_labels(_generation_group_spec(echo_schema, "fit")) == {
        "fit_method": "method"
    }


def test_generation_persistence_uses_flat_logical_keys():
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field("drive_gain_mode", "fixed")
    schema.set_field("physical_recovery_mode", "fail_triggered_fit")
    schema.set_field("earlystop_snr", 12.5)
    schema.set_field("acquire_retry", 2)

    raw = schema.to_persisted_raw()

    generation = raw["generation"]
    assert isinstance(generation, dict)
    assert "feedback" not in generation
    assert "safety" not in generation
    assert generation["drive_gain_mode"] == {"__kind": "direct", "value": "fixed"}
    assert generation["physical_recovery_mode"] == {
        "__kind": "direct",
        "value": "fail_triggered_fit",
    }
    assert "physical_recovery_max_center_shift_mhz" not in generation
    assert generation["earlystop_snr"] == {"__kind": "direct", "value": 12.5}
    assert generation["acquire_retry"] == {"__kind": "direct", "value": 2}

    restored = QubitFreqBuilder().make_default_schema()
    restored.restore_persisted_raw(raw)

    knobs = restored.read_knobs()
    assert knobs["drive_gain_mode"] == "fixed"
    assert knobs["physical_recovery_mode"] == "fail_triggered_fit"
    assert "physical_recovery_max_center_shift_mhz" not in knobs
    assert knobs["earlystop_snr"] == pytest.approx(12.5)
    assert knobs["acquire_retry"] == 2
    assert read_value_tree(restored)["generation"]["drive_gain"]["drive_gain_mode"] == (
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


def test_generation_restore_rejects_removed_feedback_enabled_key():
    schema = QubitFreqBuilder().make_default_schema()
    raw = schema.to_persisted_raw()
    generation = raw["generation"]
    assert isinstance(generation, dict)
    generation["pred_freq_correction_enabled"] = {
        "__kind": "direct",
        "value": False,
    }

    with pytest.raises(
        NodeCfgPersistenceError,
        match=r"Unknown persisted generation key\(s\): pred_freq_correction_enabled",
    ):
        QubitFreqBuilder().make_default_schema().restore_persisted_raw(raw)


@pytest.mark.parametrize(
    ("builder", "removed_key"),
    (
        (QubitFreqBuilder(), "physical_recovery_min_points"),
        (QubitFreqBuilder(), "physical_recovery_max_points"),
        (QubitFreqBuilder(), "physical_recovery_max_center_shift_mhz"),
        (QubitFreqBuilder(), "physical_recovery_max_rms_mhz"),
        (LenRabiBuilder(), "pi_product_factor"),
        (LenRabiBuilder(), "pi_gain_feedback_step_gain"),
        (LenRabiBuilder(), "pi_gain_feedback_decay_points"),
    ),
)
def test_generation_restore_rejects_removed_hard_coded_keys(
    builder: Builder, removed_key: str
):
    raw = builder.make_default_schema().to_persisted_raw()
    generation = raw["generation"]
    assert isinstance(generation, dict)
    generation[removed_key] = {"__kind": "direct", "value": 1}

    with pytest.raises(
        NodeCfgPersistenceError,
        match=rf"Unknown persisted generation key\(s\): {removed_key}",
    ):
        builder.make_default_schema().restore_persisted_raw(raw)


def test_grouped_generation_section_is_removed_from_lower_raw():
    schema = path_node_schema(
        (
            node_path(
                "drive_gain_mode",
                "generation.drive_gain.drive_gain_mode",
                str_choice_spec("drive_gain_mode", ("adaptive", "fixed")),
                "adaptive",
            ),
            node_path("reps", "reps", IntSpec("reps"), 1000),
        ),
        section_labels={
            "generation": "Generation overrides",
            "generation.drive_gain": "Drive-gain adaptation",
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
    assert read_value_tree(schema)["modules"]["qub_pulse"]["gain"] == {
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

    knobs = read_value_tree(schema)

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


# --- 2. defaults: schema/helper-derived invariants, not copied default tables ---


@pytest.mark.parametrize("builder", _BUILDERS, ids=_BUILDER_IDS)
def test_real_builder_default_lowering_is_schema_projection(builder: Builder):
    schema = builder.make_default_schema()

    knobs = schema.lower(None)
    raw = schema.lower_raw(ModuleLibrary(), md=MetaDict())

    assert set(knobs) <= set(schema.keys), builder.name
    assert {"generation", "modules", "sweep"}.isdisjoint(knobs)
    assert "generation" not in raw
    assert set(knobs).isdisjoint(_DERIVED_FORBIDDEN), builder.name


@pytest.mark.parametrize(
    "builder",
    (
        QubitFreqBuilder(),
        LenRabiBuilder(),
        T1Builder(),
        T2RamseyBuilder(),
        T2EchoBuilder(),
    ),
    ids=("qubit_freq", "lenrabi", "t1", "t2ramsey", "t2echo"),
)
def test_default_earlystop_snr_optional_clear_omits_key(builder: Builder):
    schema = builder.make_default_schema()

    assert "earlystop_snr" in schema.lower(None)
    schema.set_field("earlystop_snr", None)

    assert "earlystop_snr" not in schema.lower(None)


def test_qubit_freq_recovery_default_knobs():
    knobs = QubitFreqBuilder().make_default_schema().lower(None)

    assert knobs["physical_recovery_mode"] == "fail_triggered_fit"
    assert "physical_recovery_min_points" not in knobs
    assert "physical_recovery_max_points" not in knobs
    assert "physical_recovery_max_center_shift_mhz" not in knobs
    assert "physical_recovery_max_rms_mhz" not in knobs
    assert knobs["pred_freq_correction_idw_k"] == 10
    assert knobs["pred_freq_correction_idw_epsilon"] == pytest.approx(1e-4)
    assert knobs["pred_freq_correction_decay_points"] == 4.0


def test_qubit_freq_default_detune_sweep_is_symmetric_100mhz_window():
    detune = QubitFreqBuilder().make_default_schema().lower(None)["detune_sweep"]

    assert float(detune.start) == pytest.approx(-50.0)
    assert float(detune.stop) == pytest.approx(50.0)
    assert int(detune.expts) == 201


@pytest.mark.parametrize(
    ("builder", "field", "invalid_value", "match"),
    (
        (QubitFreqBuilder(), "qub_ch", None, r"modules\.qub_pulse\.ch"),
        (QubitFreqBuilder(), "qub_nqz", 0, r"modules\.qub_pulse\.nqz.*choices"),
        (LenRabiBuilder(), "qub_ch", None, r"modules\.rabi_pulse\.ch"),
        (LenRabiBuilder(), "qub_nqz", 0, r"modules\.rabi_pulse\.nqz.*choices"),
        (MistBuilder(), "mist_ch", None, r"modules\.mist_pulse\.ch"),
        (MistBuilder(), "mist_nqz", 0, r"modules\.mist_pulse\.nqz.*choices"),
    ),
    ids=(
        "qubit_freq_ch",
        "qubit_freq_nqz",
        "lenrabi_ch",
        "lenrabi_nqz",
        "mist_ch",
        "mist_nqz",
    ),
)
def test_default_required_routing_fields_fast_fail(
    builder: Builder, field: str, invalid_value: object, match: str
):
    schema = builder.make_default_schema()

    schema.set_field(field, invalid_value)

    with pytest.raises(RuntimeError, match=match):
        schema.lower(None)


def test_ro_optimize_default_sweep_specs_are_user_facing_ranges():
    spec = RoOptimizeBuilder().make_default_schema().schema.spec
    sweep = spec.fields["sweep"]

    assert isinstance(sweep, CfgSectionSpec)
    assert isinstance(sweep.fields["freq"], SweepSpec)
    assert isinstance(sweep.fields["gain"], SweepSpec)


def test_ro_optimize_fresh_gain_fallback_is_low_power_window():
    ro = RoOptimizeBuilder().make_default_schema().lower(None)

    assert ro["gain_half_width"] == pytest.approx(0.1)
    _assert_sweep_bounds(ro["gain_range"], (0.0, 0.2))


def test_t2echo_fresh_default_uses_recommended_detune_and_auto_fit():
    echo = T2EchoBuilder().make_default_schema().lower(None)

    assert echo["detune_ratio"] == pytest.approx(0.1)
    assert echo["fit_method"] == "auto_by_detune"


def test_operator_facing_defaults_golden_subset():
    qf = QubitFreqBuilder().make_default_schema().lower(None)
    ro = RoOptimizeBuilder().make_default_schema().lower(None)
    echo = T2EchoBuilder().make_default_schema().lower(None)

    assert qf["acquire_retry"] == 3
    assert qf["physical_recovery_mode"] == "fail_triggered_fit"
    assert "physical_recovery_min_points" not in qf
    assert "physical_recovery_max_points" not in qf
    assert "physical_recovery_max_center_shift_mhz" not in qf
    assert "physical_recovery_max_rms_mhz" not in qf
    assert qf["drive_gain_mode"] == "adaptive"

    assert ro["gain_half_width"] == pytest.approx(0.1)
    _assert_sweep_bounds(ro["gain_range"], (0.0, 0.2))

    assert echo["detune_ratio"] == pytest.approx(0.1)
    assert echo["fit_method"] == "auto_by_detune"

    for builder in _BUILDERS:
        base_cfg = builder.make_default_schema().lower_raw(
            ModuleLibrary(),
            md=MetaDict(),
        )
        assert "reset" not in base_cfg.get("modules", {}), builder.name


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
    assert lenrabi["relax_delay"] == pytest.approx(
        auto_relax_delay_from_t1(
            md.t1,
            factor=lenrabi["relax_factor"],
            minimum=lenrabi["relax_min_us"],
        )
    )
    _assert_sweep_bounds(
        lenrabi["sweep_range"],
        auto_stop_sweep_range(
            md.pi_len,
            start=lenrabi["sweep_start_us"],
            stop_factor=lenrabi["sweep_stop_factor"],
            stop_min=lenrabi["sweep_stop_min_us"],
        ),
    )

    ro = RoOptimizeBuilder().make_default_schema(ctx).lower(None, md=md)
    assert ro["t1_seed_us"] == 12.0
    assert ro["relax_delay"] == pytest.approx(
        auto_relax_delay_from_t1(
            md.t1,
            factor=ro["relax_factor"],
            minimum=None,
        )
    )
    ro_freq_seed = seed_readout_freq(ctx, fallback=0.0)
    _assert_sweep_bounds(
        ro["freq_range"],
        (
            ro_freq_seed - float(ro["freq_half_width_mhz"]),
            ro_freq_seed + float(ro["freq_half_width_mhz"]),
        ),
    )

    t1 = T1Builder().make_default_schema(ctx).lower(None, md=md)
    assert t1["t1_seed_us"] == 12.0
    assert t1["relax_delay"] == pytest.approx(
        auto_relax_delay_from_t1(
            md.t1,
            factor=t1["relax_factor"],
            minimum=t1["relax_min_us"],
        )
    )
    _assert_sweep_bounds(
        t1["sweep_range"],
        auto_stop_sweep_range(
            md.t1,
            start=t1["sweep_start_us"],
            stop_factor=t1["sweep_stop_factor"],
            stop_min=t1["sweep_stop_min_us"],
        ),
    )

    ramsey = T2RamseyBuilder().make_default_schema(ctx).lower(None, md=md)
    echo = T2EchoBuilder().make_default_schema(ctx).lower(None, md=md)
    assert ramsey["t1_seed_us"] == 12.0
    assert ramsey["t2r_seed_us"] == 8.0
    assert ramsey["relax_delay"] == pytest.approx(
        auto_relax_delay_from_t1(
            md.t1,
            factor=ramsey["relax_factor"],
            minimum=ramsey["relax_min_us"],
        )
    )
    _assert_sweep_bounds(
        ramsey["sweep_range"],
        auto_stop_sweep_range(
            md.t2r,
            start=ramsey["sweep_start_us"],
            stop_factor=ramsey["sweep_stop_factor"],
            stop_min=None,
        ),
    )
    assert echo["t1_seed_us"] == 12.0
    assert echo["t2e_seed_us"] == 9.0
    assert echo["relax_delay"] == pytest.approx(
        auto_relax_delay_from_t1(
            md.t1,
            factor=echo["relax_factor"],
            minimum=echo["relax_min_us"],
        )
    )
    _assert_sweep_bounds(
        echo["sweep_range"],
        auto_stop_sweep_range(
            md.t2e,
            start=echo["sweep_start_us"],
            stop_factor=echo["sweep_stop_factor"],
            stop_min=None,
        ),
    )

    mist = MistBuilder().make_default_schema(ctx).lower(None, md=md)
    assert mist["mist_freq"] == seed_readout_freq(ctx, fallback=0.0)


def test_fresh_node_defaults_seed_from_ml_modules():
    ml = ModuleLibrary()
    ml.register_waveform(
        qub_flat={
            "style": "flat_top",
            "length": 2.0,
            "raise_waveform": {"style": "cosine", "length": 0.02},
        }
    )
    pi_amp = {
        "type": "pulse",
        "ch": 5,
        "nqz": 1,
        "freq": 5100.0,
        "gain": 0.6,
        "waveform": {"style": "const", "length": 0.24},
    }
    readout_dpm = {
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
    }
    ml.register_module(pi_amp=pi_amp, readout_dpm=readout_dpm)
    ctx = _ctx(ml=ml)

    lenrabi = LenRabiBuilder().make_default_schema(ctx).lower(ml, md=ctx.md)
    assert lenrabi["expected_pi_length"] == pulse_length(pi_amp)
    assert lenrabi["pi_product_seed"] == pytest.approx(pulse_product(pi_amp))
    expected_pi_length = pulse_length(pi_amp)
    assert expected_pi_length is not None
    _assert_sweep_bounds(
        lenrabi["sweep_range"],
        auto_stop_sweep_range(
            expected_pi_length,
            start=lenrabi["sweep_start_us"],
            stop_factor=lenrabi["sweep_stop_factor"],
            stop_min=lenrabi["sweep_stop_min_us"],
        ),
    )

    ro = RoOptimizeBuilder().make_default_schema(ctx).lower(ml, md=ctx.md)
    ro_freq_seed = seed_readout_freq(ctx, fallback=0.0)
    _assert_sweep_bounds(
        ro["freq_range"],
        (
            ro_freq_seed - float(ro["freq_half_width_mhz"]),
            ro_freq_seed + float(ro["freq_half_width_mhz"]),
        ),
    )
    ro_gain_seed = seed_readout_gain(ctx, fallback=0.0)
    _assert_sweep_bounds(
        ro["gain_range"],
        (
            max(0.0, ro_gain_seed - float(ro["gain_half_width"])),
            min(1.0, ro_gain_seed + float(ro["gain_half_width"])),
        ),
    )


# --- 2b. equivalence: default-knob make_cfg follows schema-lowered cfg ----
#
# The defaults live in the schema; expected values come from the same schema
# lowering path, while assertions still pin runtime-injected fields separately.


def test_qubit_freq_make_cfg_uses_schema_defaults():
    ml = _ml()
    builder = QubitFreqBuilder()
    env = RunEnv(
        flux=0.0,
        flux_idx=0,
        schema=builder.make_default_schema(),
        ml=ml,
    )
    dependency_readout = {
        "type": "readout/pulse",
        "pulse_cfg": {
            "ch": 7,
            "nqz": 1,
            "freq": 6200.5,
            "gain": 0.73,
            "waveform": {"style": "const", "length": 1.7},
        },
        "ro_cfg": {
            "ro_ch": 3,
            "ro_freq": 6200.5,
            "ro_length": 1.7,
            "trig_offset": 9.0,
        },
    }
    snap = Snapshot(
        {"predict_freq": 5135.0, "qfw_factor": None},
        modules={"readout": dependency_readout},
    )
    cfg = builder.make_cfg(env, snap)
    expected_raw = env.schema.lower_raw(ml, md=MetaDict())
    expected_pulse = expected_raw["modules"]["qub_pulse"]
    expected_readout = expected_raw["modules"]["readout"]

    assert cfg.reps == expected_raw["reps"]
    assert cfg.rounds == expected_raw["rounds"]
    assert cfg.relax_delay == expected_raw["relax_delay"]
    assert int(cfg.modules.qub_pulse.ch) == int(expected_pulse["ch"])
    assert int(cfg.modules.qub_pulse.nqz) == int(expected_pulse["nqz"])
    assert float(cfg.modules.qub_pulse.gain) == pytest.approx(
        float(expected_pulse["gain"])
    )
    assert float(cfg.modules.qub_pulse.freq) == 5135.0  # the injected predict_freq
    assert isinstance(cfg.modules.readout, PulseReadoutCfg)
    assert int(cfg.modules.readout.pulse_cfg.ch) == int(
        expected_readout["pulse_cfg"]["ch"]
    )
    assert int(cfg.modules.readout.pulse_cfg.nqz) == int(
        expected_readout["pulse_cfg"]["nqz"]
    )
    assert int(cfg.modules.readout.ro_cfg.ro_ch) == int(
        expected_readout["ro_cfg"]["ro_ch"]
    )
    assert float(cfg.modules.readout.ro_cfg.trig_offset) == pytest.approx(
        float(expected_readout["ro_cfg"]["trig_offset"])
    )
    assert float(cfg.modules.readout.pulse_cfg.freq) == pytest.approx(6200.5)
    assert float(cfg.modules.readout.pulse_cfg.gain) == pytest.approx(0.73)
    assert float(cfg.modules.readout.pulse_cfg.waveform.length) == pytest.approx(1.7)
    assert float(cfg.modules.readout.ro_cfg.ro_freq) == pytest.approx(6200.5)
    assert float(cfg.modules.readout.ro_cfg.ro_length) == pytest.approx(1.7)


def test_qubit_freq_make_cfg_uses_smoothed_qfw_factor_for_drive_gain():
    ml = _ml()
    builder = QubitFreqBuilder()
    env = RunEnv(
        flux=0.0,
        flux_idx=1,
        schema=builder.make_default_schema(),
        ml=ml,
    )
    knobs = env.schema.lower(ml)

    qfw_factor = 65.0
    cfg = builder.make_cfg(
        env,
        Snapshot(
            {"predict_freq": 5135.0, "qfw_factor": qfw_factor},
            modules={"readout": _READOUT},
        ),
    )
    assert float(cfg.modules.qub_pulse.gain) == pytest.approx(
        min(1.0, float(knobs["target_kappa"]) / qfw_factor)
    )

    clamped_qfw_factor = 3.0
    clamped = builder.make_cfg(
        env,
        Snapshot(
            {"predict_freq": 5135.0, "qfw_factor": clamped_qfw_factor},
            modules={"readout": _READOUT},
        ),
    )
    assert float(clamped.modules.qub_pulse.gain) == pytest.approx(
        min(
            1.0,
            float(knobs["target_kappa"]) / clamped_qfw_factor,
        )
    )


def test_qubit_freq_make_cfg_seeds_qfw_factor_from_md_width_and_default_gain():
    ml = _ml()
    builder = QubitFreqBuilder()
    schema = builder.make_default_schema().with_overrides(
        {
            "qf_width_seed": 20.0,
            "qub_gain": 0.2,
        }
    )
    env = RunEnv(flux=0.0, flux_idx=0, schema=schema, ml=ml)
    knobs = schema.lower(ml)

    cfg = builder.make_cfg(
        env,
        Snapshot(
            {"predict_freq": 5135.0, "qfw_factor": None},
            modules={"readout": _READOUT},
        ),
    )

    seeded_qfw_factor = float(knobs["qf_width_seed"]) / float(knobs["qub_gain"])
    assert float(cfg.modules.qub_pulse.gain) == pytest.approx(
        min(1.0, float(knobs["target_kappa"]) / seeded_qfw_factor)
    )


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
    expected_raw = env.schema.lower_raw(env.ml, md=MetaDict())
    expected_waveform = expected_raw["modules"]["qub_pulse"]["waveform"]

    assert cfg.modules.qub_pulse.waveform.style == "const"
    assert float(cfg.modules.qub_pulse.waveform.length) == pytest.approx(
        float(expected_waveform["length"])
    )


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
    expected_raw = env.schema.lower_raw(env.ml, md=MetaDict())
    expected_waveform = expected_raw["modules"]["rabi_pulse"]["waveform"]

    assert cfg.modules.rabi_pulse.waveform.style == "const"
    assert float(cfg.modules.rabi_pulse.waveform.length) == pytest.approx(
        float(expected_waveform["length"])
    )


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
    expected_raw = env.schema.lower_raw(ml, md=MetaDict())
    expected_pulse = expected_raw["modules"]["mist_pulse"]

    assert cfg.reps == expected_raw["reps"]
    assert cfg.rounds == expected_raw["rounds"]
    assert cfg.relax_delay == expected_raw["relax_delay"]
    assert int(cfg.modules.mist_pulse.ch) == int(expected_pulse["ch"])
    assert float(cfg.modules.mist_pulse.gain) == pytest.approx(
        float(expected_pulse["gain"])
    )
    assert float(cfg.modules.mist_pulse.freq) == pytest.approx(
        float(expected_pulse["freq"])
    )
    assert int(cfg.modules.mist_pulse.nqz) == int(expected_pulse["nqz"])


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
    expected_raw = env.schema.lower_raw(env.ml, md=MetaDict())
    expected_waveform = expected_raw["modules"]["mist_pulse"]["waveform"]

    assert cfg.modules.mist_pulse.waveform.style == "const"
    assert float(cfg.modules.mist_pulse.waveform.length) == pytest.approx(
        float(expected_waveform["length"])
    )


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
    assert read_value_tree(schema)["modules"]["qub_pulse"]["value"]["gain"] == {
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
    assert read_value_tree(schema)["sweep"]["freq"]["start"] == {
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


# --- 2e. run-time override plan contract ---------------------------------------


def test_override_plan_serializes_and_validates_base_cfg_leaf_paths():
    plan = OverridePlan(
        (
            OverridePath(
                "modules.qub_pulse.gain",
                "after_first_point",
                "generation.drive_gain.drive_gain",
                "adaptive qubit drive",
            ),
        )
    )
    base_cfg = {
        "modules": {"qub_pulse": {"gain": 0.1, "waveform": {"length": 5.0}}},
        "reps": 1000,
    }

    validate_override_plan_base_cfg(plan, base_cfg, node_name="qubit_freq")

    assert override_plan_to_wire(plan) == [
        {
            "path": "modules.qub_pulse.gain",
            "mode": "after_first_point",
            "source": "generation.drive_gain.drive_gain",
            "reason": "adaptive qubit drive",
        }
    ]


def test_override_plan_rejects_ambiguous_or_absent_paths():
    base_cfg = {"modules": {"qub_pulse": {"gain": 0.1}}, "reps": 1000}

    with pytest.raises(ValueError, match="duplicate override paths"):
        OverridePlan(
            (
                OverridePath("reps", "all_points", "generation.test", "first"),
                OverridePath("reps", "all_points", "generation.test", "second"),
            )
        )
    with pytest.raises(ValueError, match="must target Default cfg"):
        OverridePath("generation.drive_gain.drive_gain", "all_points", "x", "y")
    with pytest.raises(ValueError, match="empty segment"):
        OverridePath("modules..gain", "all_points", "x", "y")
    with pytest.raises(ValueError, match="unsupported override mode"):
        OverridePath("reps", cast(Any, "sometimes"), "x", "y")
    with pytest.raises(ValueError, match="source must be non-empty"):
        OverridePath("reps", "all_points", "", "y")

    missing = OverridePlan(
        (OverridePath("modules.qub_pulse.freq", "all_points", "x", "y"),)
    )
    with pytest.raises(ValueError, match="absent from run-start base_cfg"):
        validate_override_plan_base_cfg(missing, base_cfg, node_name="qubit_freq")

    whole_module = OverridePlan(
        (OverridePath("modules.qub_pulse", "all_points", "x", "y"),)
    )
    with pytest.raises(ValueError, match="whole-module replacement is not allowed"):
        validate_override_plan_base_cfg(whole_module, base_cfg, node_name="qubit_freq")

    modules_root = OverridePlan((OverridePath("modules", "all_points", "x", "y"),))
    with pytest.raises(ValueError, match="whole-module replacement is not allowed"):
        validate_override_plan_base_cfg(modules_root, base_cfg, node_name="qubit_freq")


def test_apply_override_patches_copies_base_and_enforces_plan_modes():
    plan = OverridePlan(
        (
            OverridePath("modules.qub_pulse.freq", "all_points", "predict_freq", "qf"),
            OverridePath(
                "modules.qub_pulse.gain",
                "after_first_point",
                "drive_gain",
                "adaptive gain",
            ),
            OverridePath(
                "modules.qub_pulse.waveform.length",
                "fallback",
                "readout module dependency",
                "fallback leaves keep the base cfg when absent",
            ),
        )
    )
    base_cfg = {
        "modules": {
            "qub_pulse": {
                "freq": 5000.0,
                "gain": 0.1,
                "waveform": {"style": "const", "length": 5.0},
            }
        },
        "reps": 1000,
    }

    first = apply_override_patches(
        base_cfg,
        plan,
        {"modules.qub_pulse.freq": 5135.0},
        flux_idx=0,
        node_name="qubit_freq",
    )
    first_modules = cast(dict[str, Any], first["modules"])
    first_qub_pulse = cast(dict[str, Any], first_modules["qub_pulse"])
    assert first_qub_pulse["freq"] == 5135.0
    first_waveform = cast(dict[str, Any], first_qub_pulse["waveform"])
    assert first_waveform["length"] == 5.0
    assert base_cfg["modules"]["qub_pulse"]["freq"] == 5000.0

    second = apply_override_patches(
        base_cfg,
        plan,
        {
            "modules.qub_pulse.freq": 5140.0,
            "modules.qub_pulse.gain": 0.2,
            "modules.qub_pulse.waveform.length": 4.0,
        },
        flux_idx=1,
        node_name="qubit_freq",
    )
    second_modules = cast(dict[str, Any], second["modules"])
    second_qub_pulse = cast(dict[str, Any], second_modules["qub_pulse"])
    assert second_qub_pulse["gain"] == 0.2
    second_waveform = cast(dict[str, Any], second_qub_pulse["waveform"])
    assert second_waveform["length"] == 4.0

    with pytest.raises(ValueError, match="undeclared override path"):
        apply_override_patches(
            base_cfg,
            plan,
            {"modules.qub_pulse.phase": 90.0},
            flux_idx=1,
            node_name="qubit_freq",
        )
    with pytest.raises(ValueError, match="missed generated override path"):
        apply_override_patches(
            base_cfg,
            plan,
            {"modules.qub_pulse.freq": 5140.0},
            flux_idx=1,
            node_name="qubit_freq",
        )
    with pytest.raises(ValueError, match="initial-only path"):
        apply_override_patches(
            base_cfg,
            plan,
            {"modules.qub_pulse.freq": 5140.0, "modules.qub_pulse.gain": 0.2},
            flux_idx=0,
            node_name="qubit_freq",
        )


def test_apply_override_patches_rejects_whole_module_replacement():
    plan = OverridePlan(
        (
            OverridePath(
                "modules.qub_pulse",
                "all_points",
                "bad_source",
                "whole module should not be generated",
            ),
        )
    )
    base_cfg = {
        "modules": {
            "qub_pulse": {
                "freq": 5000.0,
                "gain": 0.1,
            }
        }
    }

    with pytest.raises(ValueError, match="whole-module replacement is not allowed"):
        apply_override_patches(
            base_cfg,
            plan,
            {"modules.qub_pulse": {"freq": 5135.0, "gain": 0.2}},
            flux_idx=1,
            node_name="qubit_freq",
        )

    root_plan = OverridePlan(
        (
            OverridePath(
                "modules",
                "all_points",
                "bad_source",
                "whole modules section should not be generated",
            ),
        )
    )
    with pytest.raises(ValueError, match="whole-module replacement is not allowed"):
        apply_override_patches(
            base_cfg,
            root_plan,
            {"modules": {"qub_pulse": {"freq": 5135.0, "gain": 0.2}}},
            flux_idx=1,
            node_name="qubit_freq",
        )


def test_real_builders_declare_nonempty_valid_override_plans():
    for builder in _BUILDERS:
        schema = builder.make_default_schema()
        base_cfg = schema.lower_raw(ModuleLibrary(), md=MetaDict())
        plan = builder.override_plan(schema)

        assert plan.paths, builder.name
        validate_override_plan_base_cfg(plan, base_cfg, node_name=builder.name)

    mist_schema = MistBuilder().make_default_schema()
    mist_base = mist_schema.lower_raw(ModuleLibrary(), md=MetaDict())
    assert set(mist_base["modules"]) == {"readout", "pi_pulse", "mist_pulse"}

    for builder in _BUILDERS:
        base_cfg = builder.make_default_schema().lower_raw(
            ModuleLibrary(), md=MetaDict()
        )
        assert "reset" not in base_cfg.get("modules", {}), builder.name


def test_real_builder_override_generation_sources_match_declared_paths():
    for builder in _BUILDERS:
        schema = builder.make_default_schema()
        generation_paths = {
            path
            for path in schema.logical_paths.values()
            if path.startswith("generation.")
        }

        for path in builder.override_plan(schema).paths:
            if path.source.startswith("generation."):
                assert path.source in generation_paths, (builder.name, path.source)


def test_real_builders_restrict_generated_readout_to_pulse_shape():
    for builder in _BUILDERS:
        schema = builder.make_default_schema()
        modules = schema.schema.spec.fields["modules"]
        assert isinstance(modules, CfgSectionSpec), builder.name
        readout = modules.fields["readout"]
        assert isinstance(readout, ModuleRefSpec), builder.name
        assert [spec.label for spec in readout.allowed] == ["Pulse Readout"]


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
