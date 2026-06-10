"""CfgSchema lowering — the experiment-coupled half of context ml writes (ADR-0006).

The session ``ContextService`` owns the ml/md write *transaction* (register +
bump + emit + persistence) but is free of the experiment cfg-tree. Lowering a
``CfgSchema`` into a concrete module/waveform cfg is experiment-coupled, so it
lives here, measure-side, and is handed to ``ContextService.apply_ml_writes`` as
the ``lower_module`` / ``lower_waveform`` callbacks. The Controller (the
``ContextWritePort`` façade) wires them in.

``MlEntryValidationError`` is raised here on a bad entry; it is defined in the
session ``context`` module (the context-domain error) and re-imported so the
existing ``except MlEntryValidationError`` sites stay valid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.session.services.context import MlEntryValidationError

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import CfgSchema
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def lower_module(schema: "CfgSchema", ml: "ModuleLibrary", md: "MetaDict") -> object:
    """Lower a Module ``CfgSchema`` (against live md/ml) into a module cfg.

    Raises MlEntryValidationError on validation / construction failure.
    """
    from zcu_tools.program.v2 import ModuleCfgFactory

    raw = schema.to_raw_dict(md, ml)
    try:
        return ModuleCfgFactory.from_raw(raw, ml=ml)
    except Exception as exc:  # noqa: BLE001 — surface as an expected validation error
        raise MlEntryValidationError(f"Invalid module configuration: {exc}") from exc


def lower_waveform(schema: "CfgSchema", ml: "ModuleLibrary", md: "MetaDict") -> object:
    """Lower a Waveform ``CfgSchema`` (against live md/ml) into a waveform cfg.

    Raises MlEntryValidationError on validation / construction failure.
    """
    from zcu_tools.program.v2 import WaveformCfgFactory

    raw = schema.to_raw_dict(md, ml)
    try:
        return WaveformCfgFactory.from_raw(raw, ml=ml)
    except Exception as exc:  # noqa: BLE001 — surface as an expected validation error
        raise MlEntryValidationError(f"Invalid waveform configuration: {exc}") from exc
