from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.cfg import (
    FloatSpec,
    SweepValue,
    module_leaf_patches,
    module_override_paths,
)
from zcu_tools.gui.app.autofluxdep.nodes.defaults import (
    PULSE_MODULE_LEAF_PATHS,
    READOUT_PULSE_MODULE_LEAF_PATHS,
    logical_generation_field,
    pulse_module_override_paths,
    pulse_module_patches,
    readout_module_override_paths,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.nodes.dependency_defaults import (
    is_lowerable_pulse_module,
    missing_info_value,
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.nodes.readout_defaults import (
    seed_readout_freq,
    seed_readout_gain,
)
from zcu_tools.gui.app.autofluxdep.nodes.timing_defaults import (
    auto_relax_delay_from_t1,
    auto_stop_sweep_range,
    fixed_sweep_range,
    seed_md_float,
    snapshot_float,
)
from zcu_tools.gui.session.types import ExpContext
from zcu_tools.meta_tool import MetaDict, ModuleLibrary
from zcu_tools.program.v2 import ModuleCfgFactory, PulseCfg


def _pulse_module() -> dict[str, object]:
    return {
        "type": "pulse",
        "ch": 1,
        "nqz": 2,
        "freq": 5135.0,
        "gain": 0.25,
        "waveform": {"style": "const", "length": 0.1},
    }


def _readout_module() -> dict[str, object]:
    return {
        "type": "readout/pulse",
        "pulse_cfg": {
            "ch": 0,
            "nqz": 2,
            "freq": 7444.6,
            "gain": 0.8,
            "waveform": {"style": "const", "length": 1.0},
        },
        "ro_cfg": {
            "ro_ch": 0,
            "ro_freq": 7444.6,
            "ro_length": 0.9,
            "trig_offset": 0.6,
        },
    }


def test_missing_dependency_defaults_return_none():
    assert missing_info_value() is None
    assert missing_module_value() is None


def test_timing_defaults_preserve_minimum_and_no_minimum_modes():
    assert auto_relax_delay_from_t1(2.0, factor=3.0, minimum=10.0) == 10.0
    assert auto_relax_delay_from_t1(2.0, factor=3.0, minimum=None) == 6.0
    assert auto_stop_sweep_range(0.2, start=0.05, stop_factor=5.0, stop_min=2.0) == (
        0.05,
        2.0,
    )
    assert auto_stop_sweep_range(0.2, start=0.05, stop_factor=5.0, stop_min=None) == (
        0.05,
        1.0,
    )


def test_snapshot_and_fixed_sweep_helpers():
    sweep = SweepValue(start=1.0, stop=4.0, expts=7)

    assert snapshot_float({"t1": None}, "t1", 12.0) == 12.0
    assert snapshot_float({"t1": 8}, "t1", 12.0) == 8.0
    assert fixed_sweep_range(sweep) == (1.0, 4.0)


def test_seed_helpers_use_md_or_fallback():
    md = MetaDict()
    md.update({"t1": 12.5, "r_f": 6123.0})
    ctx = ExpContext(md=md, ml=ModuleLibrary(), soc=None, soccfg=None)

    assert seed_md_float(ctx, "t1", 10.0) == 12.5
    assert seed_md_float(ctx, "missing", 10.0) == 10.0
    assert seed_readout_freq(ctx, 6000.0) == 6123.0
    assert seed_readout_gain(ctx, 0.5) == 0.5


def test_logical_generation_field_reuses_key_for_logical_and_field_path():
    field = logical_generation_field(
        "relax_factor",
        FloatSpec(label="relax_factor"),
        3.0,
        group="timing",
    )

    assert field.logical_key == "relax_factor"
    assert field.field_key == "relax_factor"
    assert field.group_key == "timing"
    assert field.group_label == "Timing / relax"


def test_module_override_helpers_preserve_paths_source_and_reason():
    assert pulse_module_override_paths(
        "pi_pulse",
        source="pi_pulse module dependency",
        reason="pi pulse is resolved from workflow/module-library dependency",
    ) == module_override_paths(
        prefix="modules.pi_pulse",
        leaf_paths=PULSE_MODULE_LEAF_PATHS,
        source="pi_pulse module dependency",
        reason="pi pulse is resolved from workflow/module-library dependency",
    )
    assert readout_module_override_paths(
        source="opt_readout module dependency",
        reason="readout module is resolved from workflow/module-library dependency",
    ) == module_override_paths(
        prefix="modules.readout",
        leaf_paths=READOUT_PULSE_MODULE_LEAF_PATHS,
        source="opt_readout module dependency",
        reason="readout module is resolved from workflow/module-library dependency",
    )


def test_module_patch_helpers_preserve_leaf_outputs():
    pulse = _pulse_module()
    readout = _readout_module()

    assert pulse_module_patches("pi_pulse", pulse) == module_leaf_patches(
        prefix="modules.pi_pulse",
        module=pulse,
        leaf_paths=PULSE_MODULE_LEAF_PATHS,
    )
    assert readout_module_patches(readout) == module_leaf_patches(
        prefix="modules.readout",
        module=readout,
        leaf_paths=READOUT_PULSE_MODULE_LEAF_PATHS,
    )


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (_pulse_module(), True),
        ({"type": "readout/pulse"}, False),
        (None, False),
    ],
)
def test_is_lowerable_pulse_module_dict_cases(module, expected):
    assert is_lowerable_pulse_module(module) is expected


def test_is_lowerable_pulse_module_accepts_pulse_cfg():
    pulse = ModuleCfgFactory.from_raw(_pulse_module())

    assert isinstance(pulse, PulseCfg)
    assert is_lowerable_pulse_module(pulse) is True
