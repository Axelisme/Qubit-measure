"""Fixed project and context tools for the measure-gui setup flow."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def project(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read or atomically update a project through the GUI owner."""
    if arguments:
        names = {
            "chip": "chip_name",
            "qubit": "qub_name",
            "resonator": "res_name",
            "scope": "scope_id",
        }
        params = {
            wire: arguments[name] for name, wire in names.items() if name in arguments
        }
        result = ctx.send_gui_rpc("startup.apply", params)
    else:
        result = ctx.session.read_internal("project.info", {})
    return {
        "chip": result["chip_name"],
        "qubit": result["qub_name"],
        "resonator": result["res_name"],
        "result_dir": result["result_dir"],
        "database_path": result["database_path"],
    }


def contexts(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """List context labels and the current selection."""
    raise NotImplementedError("03 contexts tool has no implementation yet")


def context_use(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Select a named GUI context."""
    raise NotImplementedError("03 context_use tool has no implementation yet")


def context_create(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Create a named or device-derived context through the GUI."""
    raise NotImplementedError("03 context_create tool has no implementation yet")


_PROJECT_FIELD = {"type": "string", "minLength": 1}

PROJECT_TOOLS: dict[str, dict[str, Any]] = {
    "project": {
        "handler": project,
        "description": (
            "Read the applied project with no arguments, or atomically update only the "
            "provided chip/qubit/resonator/scope in the GUI. Missing identity fields "
            "on a first update fail. Changing chip/qubit without scope selects the "
            "new identity's GUI default scope; an effective change deactivates the "
            "active context. An equal write or failed update does not. Returns "
            "{chip, qubit, resonator, result_dir, database_path}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chip": _PROJECT_FIELD,
                "qubit": _PROJECT_FIELD,
                "resonator": _PROJECT_FIELD,
                "scope": _PROJECT_FIELD,
            },
        },
    },
    "contexts": {
        "handler": contexts,
        "description": "List GUI context labels and the active label as {active, labels}; no context is active after a project change.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "context_use": {
        "handler": context_use,
        "description": "Select an existing context label; unknown labels fail with available choices. Returns {label}.",
        "inputSchema": {
            "type": "object",
            "properties": {"label": _PROJECT_FIELD},
            "required": ["label"],
        },
    },
    "context_create": {
        "handler": context_create,
        "description": (
            "Create and select a context. Optional label names it; without label, "
            "bind_device supplies the current device value and unit for a default "
            "label. clone_from defaults to the active context (or empty when none); "
            "explicit null starts empty. Returns {label}. Invalid source leaves the "
            "prior active context and labels unchanged."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": _PROJECT_FIELD,
                "bind_device": _PROJECT_FIELD,
                "clone_from": {
                    "type": ["string", "null"],
                    "default": "current",
                },
            },
        },
    },
}


def build_setup_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in PROJECT_TOOLS.items()
    }
