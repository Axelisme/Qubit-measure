"""Fixed operation tools over the GUI-owned operation and session state."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def status(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read current GUI orientation and every live GUI-owned operation."""
    del arguments
    session = ctx.session
    has_project = bool(session.read_internal("state.has_project", {})["value"])
    has_context = bool(session.read_internal("state.has_active_context", {})["value"])
    has_soc = bool(session.read_internal("state.has_soc", {})["value"])

    project: dict[str, Any] | None = None
    if has_project:
        info = session.read_internal("project.info", {})
        project = {
            "chip": info["chip_name"],
            "qubit": info["qub_name"],
            "resonator": info["res_name"],
        }
    soc = {"connected": has_soc, "mock": False}
    if has_soc:
        soc["mock"] = bool(session.read_internal("soc.info", {})["is_mock"])

    tabs = [
        {
            "tab": tab["tab_id"],
            "experiment": tab["adapter_name"],
            "running": bool(tab["interaction"]["is_running"]),
        }
        for tab in session.read_internal("tab.snapshot", {})["tabs"]
    ]
    missing = []
    if not has_project:
        missing.append("project")
    if not has_context:
        missing.append("active_context")
    if not has_soc:
        missing.append("soc")
    if not tabs:
        missing.append("tab")
    return {
        "project": project,
        "soc": soc,
        "context": {
            "active": session.read_internal("context.active", {})["label"]
        },
        "devices": [
            {"name": device["name"], "connected": device["status"] == "connected"}
            for device in session.read_internal("device.list", {})["devices"]
        ],
        "predictor": {
            "loaded": session.read_internal("predictor.info", {})["loaded"]
        },
        "ready": {"can_run": not missing, "missing": missing},
        "tabs": tabs,
        "running": session.read_internal("operation.active", {})["operations"],
    }


def wait(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Wait on a known GUI operation, reporting its outcome as data."""
    del ctx, arguments
    raise NotImplementedError("wait projection is not implemented")


def cancel(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Request cancellation of a known cancellable GUI operation."""
    del ctx, arguments
    raise NotImplementedError("cancel projection is not implemented")


def build_operation_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "status": {
            "handler": partial(status, ctx),
            "description": "Index the live GUI session and all in-flight operations.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        "wait": {
            "handler": partial(wait, ctx),
            "description": "Wait for an operation, or return running at the timeout.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op": {"type": "integer"},
                    "timeout": {"type": "number", "default": 60, "minimum": 0, "maximum": 300},
                },
                "required": ["op"],
            },
        },
        "cancel": {
            "handler": partial(cancel, ctx),
            "description": "Cancel a known operation when it has a cancel hook.",
            "inputSchema": {
                "type": "object", "properties": {"op": {"type": "integer"}}, "required": ["op"]
            },
        },
    }
