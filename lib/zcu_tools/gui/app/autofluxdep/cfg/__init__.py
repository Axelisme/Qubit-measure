"""Local cfg seam for autofluxdep nodes — the SINGLE allowed import point of the
measure-app spec/value model (``gui.app.main.adapter``).

autofluxdep reuses the framework's typed cfg machinery (the pure, experiment-free
spec/value tree + ``CfgSchema`` lowering, ADR-0011) to type each node's user
knobs, mirroring measure-gui's cfg editor. That machinery currently lives under
``gui/app/main/adapter`` (the measure app). Re-exporting it here — and only here —
keeps the app-to-app coupling at one seam: a future lift of the spec/value model
into a shared layer (``gui/session/cfg`` etc.) only has to retarget this file's
imports, not every node.

The node-facing helpers build explicit path-mounted schemas with stable
logical-key projection. Node cfg still lowers through the shared spec/value
model while autofluxdep builders keep stable logical knob names.
"""

from __future__ import annotations

# re-exported framework spec/value model — pure data, no experiment knowledge
from zcu_tools.gui.app.main.adapter import (
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
    ScalarSpec,
    SweepSpec,
    SweepValue,
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
    "CfgSectionSpec",
    "CfgSectionValue",
    "ChoiceBinding",
    "ChoiceSectionSpec",
    "DirectValue",
    "EvalValue",
    "FloatSpec",
    "IntSpec",
    "ModuleRefSpec",
    "NodeCfgSchema",
    "OverrideMode",
    "OverridePath",
    "OverridePlan",
    "RunCfgSnapshot",
    "ScalarSpec",
    "SweepSpec",
    "SweepValue",
    "apply_override_patches",
    "empty_node_schema",
    "module_leaf_patches",
    "module_override_paths",
    "override_plan_to_wire",
    "str_choice_spec",
    "validate_override_plan_base_cfg",
]
