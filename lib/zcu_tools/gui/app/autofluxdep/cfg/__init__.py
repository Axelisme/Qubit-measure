"""Local cfg seam for autofluxdep node schemas and app-owned conversions.

The generic spec/value model, inheritance helpers, and persistence codec come
from ``zcu_tools.gui.cfg``. Pulse/readout spec construction and module conversion
remain autoflux app responsibilities.
"""

from __future__ import annotations

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
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    align_locked_literals,
    make_default_value,
)

from .module_adapter import (
    pulse_module_ref_spec,
    pulse_readout_module_ref_spec,
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
    "NodeCfgSchema",
    "OverrideMode",
    "OverridePath",
    "OverridePlan",
    "RunCfgSnapshot",
    "ScalarSpec",
    "SweepSpec",
    "SweepValue",
    "ReferenceSpec",
    "ReferenceValue",
    "align_locked_literals",
    "apply_override_patches",
    "empty_node_schema",
    "make_default_value",
    "module_leaf_patches",
    "module_override_paths",
    "override_plan_to_wire",
    "pulse_module_ref_spec",
    "pulse_readout_module_ref_spec",
    "str_choice_spec",
    "validate_override_plan_base_cfg",
]
