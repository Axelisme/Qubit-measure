"""Local cfg seam for autofluxdep node schemas and measure-owned conversions.

The generic spec/value model, inheritance helpers, and persistence codec come
from ``zcu_tools.gui.cfg``. Pulse/readout spec construction and module conversion
remain measure-app responsibilities in this slice.
"""

from __future__ import annotations

from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec
from zcu_tools.gui.app.main.specs.readout import make_pulse_readout_spec

# Re-exported shared spec/value model — pure data, no experiment knowledge.
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    FloatSpec,
    IntSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    align_locked_literals,
    make_default_value,
)

from .override_plan import (
    OverrideMode,
    OverridePath,
    OverridePlan,
    RunCfgSnapshot,
    apply_override_patches,
    module_leaf_patches,
    module_override_paths,
    override_plan_to_wire,
    validate_override_plan_base_cfg,
)
from .schema import (
    NodeCfgSchema,
    empty_node_schema,
    str_choice_spec,
)


def pulse_module_ref_spec(
    label: str = "Pulse", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[make_pulse_spec()],
        label=label,
        optional=optional,
    )


def pulse_readout_module_ref_spec(
    label: str = "Readout", optional: bool = False
) -> ModuleRefSpec:
    return ModuleRefSpec(
        allowed=[make_pulse_readout_spec()],
        label=label,
        optional=optional,
    )


__all__ = [
    "CfgSchema",
    "CfgNodeSpec",
    "CfgSectionSpec",
    "CfgSectionValue",
    "CenteredSweepSpec",
    "CenteredSweepValue",
    "ChoiceBinding",
    "ChoiceSectionSpec",
    "DirectValue",
    "EvalValue",
    "FloatSpec",
    "IntSpec",
    "ModuleRefSpec",
    "ModuleRefValue",
    "NodeCfgSchema",
    "OverrideMode",
    "OverridePath",
    "OverridePlan",
    "RunCfgSnapshot",
    "ScalarSpec",
    "SweepSpec",
    "SweepValue",
    "WaveformRefSpec",
    "WaveformRefValue",
    "align_locked_literals",
    "apply_override_patches",
    "empty_node_schema",
    "make_default_value",
    "module_leaf_patches",
    "module_cfg_to_value",
    "module_override_paths",
    "override_plan_to_wire",
    "pulse_module_ref_spec",
    "pulse_readout_module_ref_spec",
    "str_choice_spec",
    "validate_override_plan_base_cfg",
]
