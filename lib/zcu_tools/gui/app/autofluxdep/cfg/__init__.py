"""Local cfg seam for autofluxdep nodes — the SINGLE allowed import point of the
measure-app spec/value model (``gui.app.main.adapter``).

autofluxdep reuses the framework's typed cfg machinery (the pure, experiment-free
spec/value tree + ``CfgSchema`` lowering, ADR-0011) to type each node's user
knobs, mirroring measure-gui's cfg editor. That machinery currently lives under
``gui/app/main/adapter`` (the measure app). Re-exporting it here — and only here —
keeps the app-to-app coupling at one seam: a future lift of the spec/value model
into a shared layer (``gui/session/cfg`` etc.) only has to retarget this file's
imports, not every node.

The node-facing spec helpers (``flat_scalar_schema`` + the ``*Spec`` sugar) build
the *flat* user-knob schema each node lowers (decision: waveforms stay by-name
string scalars + flat scalar pulse fields, no ModuleRef/WaveformRef — that keeps
ModuleLibrary the SSOT and ``make_cfg``'s ``ml.get_waveform`` path intact).
"""

from __future__ import annotations

# re-exported framework spec/value model — pure data, no experiment knowledge
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    FloatSpec,
    IntSpec,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)

from .schema import (
    NodeCfgSchema,
    flat_node_schema,
    str_scalar_spec,
)

__all__ = [
    "CfgSchema",
    "CfgSectionSpec",
    "CfgSectionValue",
    "DirectValue",
    "FloatSpec",
    "IntSpec",
    "NodeCfgSchema",
    "ScalarSpec",
    "SweepSpec",
    "SweepValue",
    "flat_node_schema",
    "str_scalar_spec",
]
