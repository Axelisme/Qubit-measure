"""Fixed operation tools over the GUI-owned operation and session state."""

from __future__ import annotations

import math
import time
from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
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
        "context": {"active": session.read_internal("context.active", {})["label"]},
        "devices": [
            {"name": device["name"], "connected": device["status"] == "connected"}
            for device in session.read_internal("device.list", {})["devices"]
        ],
        "predictor": {"loaded": session.read_internal("predictor.info", {})["loaded"]},
        "ready": {"can_run": not missing, "missing": missing},
        "tabs": tabs,
        "running": [
            {**operation, "op": session.expose_operation(operation["op"])}
            for operation in session.read_internal("operation.active", {})["operations"]
        ],
    }


def _operation_id(arguments: dict[str, Any]) -> int:
    op = arguments.get("op")
    if isinstance(op, bool) or not isinstance(op, int) or op <= 0:
        raise ValueError("op must be a positive integer")
    return op


def wait(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Wait on a known GUI operation. elapsed_s measures this wait call."""
    op = _operation_id(arguments)
    timeout = arguments.get("timeout", 60)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 <= timeout <= 300
    ):
        raise ValueError("timeout must be between 0 and 300 seconds")
    start = time.monotonic()
    reply = ctx.send_gui_rpc(
        "operation.await",
        {"timeout": timeout},
        timeout_seconds=float(timeout) + 2.0,
        operation_handle=op,
    )
    result: dict[str, Any] = {"elapsed_s": max(0.0, time.monotonic() - start)}
    if reply["reason"] == "completed":
        result["status"] = reply["status"]
        if "error" in reply:
            result["error"] = reply["error"]
        if "feedback" in reply:
            result["feedback"] = reply["feedback"]
        return result
    if reply["reason"] not in ("timeout", "user_feedback"):
        raise ValueError(f"unexpected operation await reason: {reply['reason']!r}")
    result["status"] = "running"
    if "feedback" in reply:
        result["feedback"] = reply["feedback"]
    progress = ctx.session.read_internal("operation.progress", {}, operation_handle=op)
    if progress["active"]:
        bars = progress["bars"]
        result["progress"] = bars
        estimates = [bar["eta_s"] for bar in bars if bar.get("eta_s") is not None]
        if estimates:
            result["eta_s"] = max(estimates)
    return result


def cancel(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Ask the GUI's domain owner to stop, then await a bounded terminal."""
    op = _operation_id(arguments)
    response = ctx.session.read_internal("operation.cancel", {}, operation_handle=op)
    if response["status"] != "cancelling":
        return {"status": response["status"]}
    outcome = wait(ctx, {"op": op, "timeout": 0.25})
    if outcome["status"] == "cancelled":
        return {"status": "cancelled"}
    if outcome["status"] == "failed":
        error = outcome.get("error", {})
        raise GuiRpcError(
            f"operation {op} failed: {error.get('message', 'unknown failure')}",
            reason="operation_failed",
            code="precondition_failed",
        )
    if outcome["status"] == "finished":
        return {"status": "finished"}
    return {"status": "cancelling"}


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
                    "timeout": {
                        "type": "number",
                        "default": 60,
                        "minimum": 0,
                        "maximum": 300,
                    },
                },
                "required": ["op"],
            },
        },
        "cancel": {
            "handler": partial(cancel, ctx),
            "description": "Cancel a known operation when it has a cancel hook.",
            "inputSchema": {
                "type": "object",
                "properties": {"op": {"type": "integer"}},
                "required": ["op"],
            },
        },
    }
