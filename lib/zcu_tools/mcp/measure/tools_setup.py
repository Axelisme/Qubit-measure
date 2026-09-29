"""Fixed project and context tools for the measure-gui setup flow."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
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
        result = ctx.send_gui_rpc("project.apply", params)
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
    if "keys" not in arguments:
        result = ctx.session.read_internal("context.md_get", {"summaries": True})
        return {"values": result["values"]}
    return {
        "values": {
            key: ctx.session.read_internal("context.md_get_attr", {"key": key})["value"]
            for key in arguments["keys"]
        }
    }


def soc_info(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read the GUI's connection and hardware projection."""
    connected = bool(ctx.session.read_internal("state.has_soc", {})["value"])
    if not connected:
        return {
            "connected": False,
            "address": None,
            "port": None,
            "description": None,
            "is_mock": False,
        }
    info = ctx.session.read_internal(
        "soc.info", {"include_cfg": arguments.get("include_cfg", False)}
    )
    return {"connected": True, **info}


def soc_connect(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Synchronously connect the GUI to a remote board and read its info."""
    ctx.send_gui_rpc(
        "soc.connect",
        {"kind": "remote", "ip": arguments["address"], "port": arguments["port"]},
    )
    return soc_info(ctx, {"include_cfg": False})


def md_set(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Write MetaDict keys in order without rolling back prior successful writes."""
    applied: dict[str, dict[str, Any]] = {}
    for key, value in arguments["values"].items():
        try:
            reply = ctx.send_gui_rpc(
                "context.md_set_attr", {"key": key, "value": value, "receipt": True}
            )
            if "before" not in reply or "after" not in reply:
                raise GuiRpcError(
                    "missing GUI MetaDict write receipt", reason="incompatible_wire"
                )
        except (GuiRpcError, OSError) as exc:
            raise GuiRpcError(
                f"md_set failed at {key!r}; confirmed prefix: {applied!r}; {exc}. "
                "The failing key may also have applied; read md_get before retrying.",
                reason=exc.reason
                if isinstance(exc, GuiRpcError)
                else "transport_error",
                code=exc.code if isinstance(exc, GuiRpcError) else None,
            ) from exc
        applied[key] = {"before": reply["before"], "after": reply["after"]}
    return applied


_PROJECT_FIELD = {"type": "string", "minLength": 1}

PROJECT_TOOLS: dict[str, dict[str, Any]] = {
    "soc_connect": {
        "handler": soc_connect,
        "description": (
            "Synchronously connect the GUI to a remote SoC at address/port. "
            "Replacing a connection uses the same GUI owner; no operation handle or "
            "automatic retry. Returns soc_info(include_cfg=false) after success."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": _PROJECT_FIELD,
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
            },
            "required": ["address", "port"],
        },
    },
    "soc_info": {
        "handler": soc_info,
        "description": (
            "Read SoC connection state and the GUI's per-channel description. "
            "Disconnected state has connected=false and null address/port; "
            "include_cfg=true adds the complete QICK cfg when connected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"include_cfg": {"type": "boolean", "default": False}},
        },
    },
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
