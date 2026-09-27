"""Fixed ModuleLibrary tool contracts over the GUI-owned context/editor."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext

_NAME = {"type": "string", "minLength": 1}
_KIND = {"type": "string", "enum": ["module", "waveform"]}
_EDITS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"path": _NAME, "value": {}},
        "required": ["path", "value"],
    },
}


def ml_get(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read the library index or one stored cfg without changing the library."""
    raise NotImplementedError


def ml_roles(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> list[dict[str, Any]]:
    """List live GUI role templates for ModuleLibrary creation."""
    raise NotImplementedError


def ml_create(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Create from a role's md-backed defaults and return the stored cfg."""
    raise NotImplementedError


def ml_edit(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Edit a disposable draft; commit only after every edit succeeds."""
    raise NotImplementedError


ML_TOOLS: dict[str, dict[str, Any]] = {
    "ml_get": {
        "handler": ml_get,
        "description": "Without name list modules/waveforms by name, kind/style and description. With name return {name, kind, cfg}; provide kind=module|waveform if both collections contain the name. A read does not open an editing draft.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": _NAME, "kind": _KIND},
        },
    },
    "ml_roles": {
        "handler": ml_roles,
        "description": "List the current GUI role templates [{role_id, label, kind, default_name}].",
        "inputSchema": {"type": "object"},
    },
    "ml_create": {
        "handler": ml_create,
        "description": "Create a module/waveform from role_id with md-backed defaults; use the role default_name if name omitted. Return {name, kind, cfg}; if no default_name exists, supply name.",
        "inputSchema": {
            "type": "object",
            "properties": {"role_id": _NAME, "name": _NAME},
            "required": ["role_id"],
        },
    },
    "ml_edit": {
        "handler": ml_edit,
        "description": "Edit an existing named ModuleLibrary item using ordered canonical {path,value} agent edits, then commit only on full success. save_as writes a new item and leaves the source unchanged; an error discards the draft without writing to the library. Eval inputs lower to numeric values at commit. Return {name,cfg}; kind disambiguates shared names.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": _NAME,
                "kind": _KIND,
                "edits": _EDITS,
                "save_as": _NAME,
            },
            "required": ["name", "edits"],
        },
    },
}


def build_ml_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in ML_TOOLS.items()
    }
