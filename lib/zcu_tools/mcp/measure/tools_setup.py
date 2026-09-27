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
    del arguments
    labels = ctx.session.read_internal("context.labels", {})["labels"]
    active = ctx.session.read_internal("context.active", {})["label"]
    return {"active": active, "labels": labels}


def context_use(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Select a named GUI context."""
    result = ctx.send_gui_rpc("context.use", {"label": arguments["label"]})
    return {"label": result["label"]}


def context_create(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Create a named or device-derived context through the GUI."""
    result = ctx.send_gui_rpc(
        "context.new",
        {
            "label": arguments.get("label"),
            "bind_device": arguments.get("bind_device"),
            "clone_from": arguments.get("clone_from", "current"),
        },
    )
    return {"label": result["label"]}


def md_get(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read selected values or the GUI's safe index of MetaDict values."""
    raise NotImplementedError("03 md_get tool has no implementation yet")


def md_set(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Write MetaDict keys in order without rolling back prior successful writes."""
    raise NotImplementedError("03 md_set tool has no implementation yet")


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
    "md_get": {
        "handler": md_get,
        "description": (
            "Read MetaDict as {values: {key: value}}. With no keys, GUI returns "
            "scalars and summaries of non-scalars, not their content. Supplying "
            "keys returns full values; missing keys fail rather than being skipped."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": _PROJECT_FIELD,
                    "uniqueItems": True,
                }
            },
        },
    },
    "md_set": {
        "handler": md_set,
        "description": (
            "Write values to GUI MetaDict in the order supplied. Returns "
            "{key: {before, after}} on success. On failure, earlier writes remain; "
            "the error includes the confirmed prefix and failing key. An ambiguous "
            "transport failure may also have applied the failing key; read before retrying."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"values": {"type": "object"}},
            "required": ["values"],
        },
    },
}


def build_setup_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in PROJECT_TOOLS.items()
    }
