"""Local cfg seam for autofluxdep nodes — the SINGLE allowed import point of the
measure-app spec/value model (``gui.app.main.adapter``).

autofluxdep reuses the framework's typed cfg machinery (the pure, experiment-free
spec/value tree + ``CfgSchema`` lowering, ADR-0011) to type each node's user
knobs, mirroring measure-gui's cfg editor. That machinery currently lives under
``gui/app/main/adapter`` (the measure app). Re-exporting it here — and only here —
keeps the app-to-app coupling at one seam: a future lift of the spec/value model
into a shared layer (``gui/session/cfg`` etc.) only has to retarget this file's
imports, not every node.

The node-facing helpers build either legacy flat schemas or sectioned schemas
with stable logical-key projection. Node cfg still uses scalar/sweep leaves only:
waveforms stay by-name string scalars, and workflow modules stay outside node cfg.
"""

from __future__ import annotations

# re-exported framework spec/value model — pure data, no experiment knowledge
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    FloatSpec,
    IntSpec,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)

from .schema import (
    NodeCfgSchema,
    NodeFieldDecl,
    NodeFieldSpec,
    NodeSectionSpec,
    flat_node_schema,
    node_field,
    node_section,
    sectioned_node_schema,
    str_scalar_spec,
)

__all__ = [
    "CfgSchema",
    "CfgSectionSpec",
    "CfgSectionValue",
    "DirectValue",
    "EvalValue",
    "FloatSpec",
    "IntSpec",
    "NodeCfgSchema",
    "NodeFieldDecl",
    "NodeFieldSpec",
    "NodeSectionSpec",
    "ScalarSpec",
    "SweepSpec",
    "SweepValue",
    "flat_node_schema",
    "node_field",
    "node_section",
    "sectioned_node_schema",
    "str_scalar_spec",
]
