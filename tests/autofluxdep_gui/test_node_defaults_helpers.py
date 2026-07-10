from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.cfg import module_leaf_patches
from zcu_tools.gui.app.autofluxdep.experiments._support.dependency_defaults import (
    is_lowerable_pulse_module,
    missing_info_value,
    missing_module_value,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.readout_defaults import (
    seed_readout_freq,
    seed_readout_gain,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.timing_defaults import (
    auto_relax_delay_from_t1,
    auto_stop_sweep_range,
    fixed_sweep_range,
    seed_md_float,
    snapshot_float,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.utils.override_plan import (
    PULSE_MODULE_LEAF_PATHS,
    READOUT_FALLBACK_LEAF_PATHS,
    NodeOverridePlan,
    pulse_module_patches,
    readout_module_patches,
)
from zcu_tools.gui.app.autofluxdep.experiments._support.utils.timing import (
    pop_sweep_range,
    pop_sweep_ranges,
)
from zcu_tools.gui.cfg import SweepValue
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
    assert auto_stop_sweep_range(
        10.0,
        start=0.05,
        stop_factor=5.0,
        stop_min=20.0,
        stop_max=30.0,
    ) == (0.05, 30.0)


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


def test_module_patch_helpers_preserve_leaf_outputs():
    pulse = _pulse_module()
    readout = _readout_module()

    assert pulse_module_patches("pi_pulse", pulse) == {
        "modules.pi_pulse.type": "pulse",
        "modules.pi_pulse.ch": 1,
        "modules.pi_pulse.nqz": 2,
        "modules.pi_pulse.freq": 5135.0,
        "modules.pi_pulse.gain": 0.25,
        "modules.pi_pulse.waveform": {"style": "const", "length": 0.1},
    }
    assert readout_module_patches(readout) == module_leaf_patches(
        prefix="modules.readout",
        module=readout,
        leaf_paths=READOUT_FALLBACK_LEAF_PATHS,
    )
    assert readout_module_patches(readout) == {
        "modules.readout.pulse_cfg.freq": 7444.6,
        "modules.readout.pulse_cfg.gain": 0.8,
        "modules.readout.pulse_cfg.waveform.length": 1.0,
        "modules.readout.ro_cfg.ro_freq": 7444.6,
        "modules.readout.ro_cfg.ro_length": 0.9,
    }


def test_module_override_plan_preserves_dependency_metadata():
    plan = (
        NodeOverridePlan()
        .pulse_module_dependency("pi_pulse")
        .readout_dependency()
        .build()
    )

    pulse_paths = [
        path for path in plan.paths if path.path.startswith("modules.pi_pulse")
    ]
    readout_paths = [
        path for path in plan.paths if path.path.startswith("modules.readout")
    ]
    assert len(pulse_paths) == len(PULSE_MODULE_LEAF_PATHS)
    assert {(path.mode, path.source, path.reason) for path in pulse_paths} == {
        (
            "all_points",
            "pi_pulse module dependency",
            "pi pulse is resolved from workflow/module-library dependency",
        )
    }
    assert len(readout_paths) == len(READOUT_FALLBACK_LEAF_PATHS)
    assert {(path.mode, path.source, path.reason) for path in readout_paths} == {
        (
            "fallback",
            "readout module dependency",
            "readout module is resolved from workflow/module-library dependency",
        )
    }


def test_pop_sweep_range_extracts_range_and_removes_sweep_section():
    raw_cfg = {
        "sweep": {"length": SweepValue(start=1.0, stop=4.0, expts=7)},
        "reps": 100,
    }

    assert pop_sweep_range(raw_cfg, "length", node_name="lenrabi") == (1.0, 4.0)
    assert raw_cfg == {"reps": 100}


def test_pop_sweep_ranges_accepts_range_objects_and_tuples():
    raw_cfg = {
        "sweep": {
            "freq": SweepValue(start=7000.0, stop=7100.0, expts=11),
            "gain": (0.1, 0.3),
        }
    }

    assert pop_sweep_ranges(raw_cfg, ("freq", "gain"), node_name="ro_optimize") == {
        "freq": (7000.0, 7100.0),
        "gain": (0.1, 0.3),
    }
    assert raw_cfg == {}


def test_pop_sweep_ranges_fast_fails_missing_section_or_key():
    with pytest.raises(RuntimeError, match="t1 raw cfg has no sweep section"):
        pop_sweep_range({}, "length", node_name="t1")

    with pytest.raises(RuntimeError, match=r"ro_optimize raw cfg has no sweep\.gain"):
        pop_sweep_ranges(
            {"sweep": {"freq": (7000.0, 7100.0)}},
            ("freq", "gain"),
            node_name="ro_optimize",
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
