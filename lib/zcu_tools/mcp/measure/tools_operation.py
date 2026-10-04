"""Fixed operation tools over the GUI-owned operation and session state."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from functools import partial
from typing import Any

from recipes import RECIPES
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.execution_reply import project_control, project_execution
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def status(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read a local execution, or GUI orientation and nonterminal executions."""
    detail = arguments.get("detail", "summary")
    if detail not in ("summary", "full"):
        raise ValueError("detail must be summary or full")
    if detail == "full" and "execution" not in arguments:
        raise ValueError("detail=full requires execution")
    if "execution" in arguments:
        key = _execution_id(arguments)
        if key.startswith("recipe-"):
            snapshot = ctx.session.recipes.get(key).snapshot()
            definition = next(
                item for item in RECIPES if item.name == snapshot["recipe"]
            )
            return project_execution(
                snapshot,
                definition=definition,
                detail=detail,
            )
        return project_execution(
            asdict(ctx.session.executions.get(key).snapshot()),
            detail=detail,
        )
    session = ctx.gui
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
    execution_snapshots = [
        asdict(item) for item in ctx.session.executions.snapshots()
    ] + ctx.session.recipes.snapshots()
    definitions = {definition.name: definition for definition in RECIPES}
    return {
        "project": project,
        "soc": soc,
        "context": {"active": session.read_internal("context.active", {})["label"]},
        "devices": [
            {**device, "connected": device["status"] == "connected"}
            for device in session.read_internal("device.list", {})["devices"]
        ],
        "predictor": {"loaded": session.read_internal("predictor.info", {})["loaded"]},
        "ready": {"can_run": not missing, "missing": missing},
        "tabs": tabs,
        "executions": [
            project_execution(
                item,
                definition=definitions[item["recipe"]] if "recipe" in item else None,
            )
            for item in execution_snapshots
            if item["phase"] != "terminal"
        ],
        "terminal_count": sum(
            item["phase"] == "terminal" for item in execution_snapshots
        ),
        "query_hint": 'Use status(execution=<id>, detail="full") for completed executions.',
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


def _execution_id(arguments: dict[str, Any]) -> str:
    execution = arguments["execution"]
    if not isinstance(execution, str) or not execution:
        raise ValueError("execution must be a non-empty string")
    return execution


def wait_timeout(arguments: dict[str, Any]) -> float:
    """Validate the shared operation/setup waiting budget before GUI access."""
    timeout = arguments.get("timeout", 60)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 <= timeout <= 300
    ):
        raise ValueError("timeout must be between 0 and 300 seconds")
    return float(timeout)


def _wait_tool(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any] | ToolReply:
    """Wait on exactly one operation or execution; timeout does not cancel it."""
    if ("op" in arguments) == ("execution" in arguments):
        raise ValueError("provide exactly one of op or execution")
    if "execution" in arguments:
        timeout = wait_timeout(arguments)
        start = time.monotonic()
        key = _execution_id(arguments)
        execution = (
            ctx.session.recipes.get(key)
            if key.startswith("recipe-")
            else ctx.session.executions.get(key)
        )
        reply = execution.wait(timeout)
        definition = (
            next(item for item in RECIPES if item.name == reply.data["recipe"])
            if key.startswith("recipe-")
            else None
        )
        data = project_execution(reply.data, definition=definition)
        return ToolReply(
            {**data, "elapsed_s": max(0.0, time.monotonic() - start)},
            reply.images,
        )
    return wait(ctx, arguments)


def wait(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Wait on a GUI operation, not its downstream analysis completion."""
    op = _operation_id(arguments)
    timeout = wait_timeout(arguments)
    ctx = ctx.bound()
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
    progress = ctx.gui.read_internal("operation.progress", {}, operation_handle=op)
    if progress["active"]:
        bars = progress["bars"]
        result["progress"] = bars
        estimates = [bar["eta_s"] for bar in bars if bar.get("eta_s") is not None]
        if estimates:
            result["eta_s"] = max(estimates)
    return result


def cancel(
    ctx: MeasureToolContext, arguments: dict[str, Any]
) -> dict[str, Any] | ToolReply:
    """Cancel one registered execution or an unregistered GUI operation."""
    if ("op" in arguments) == ("execution" in arguments):
        raise ValueError("provide exactly one of op or execution")
    if "execution" in arguments:
        key = _execution_id(arguments)
        execution = (
            ctx.session.recipes.get(key)
            if key.startswith("recipe-")
            else ctx.session.executions.get(key)
        )
    else:
        op = _operation_id(arguments)
        execution = ctx.session.recipes.for_op(op) or ctx.session.executions.for_op(op)
    if execution is not None:
        reply = execution.cancel()
        return ToolReply(project_control(reply.data), is_error=reply.is_error)
    op = _operation_id(arguments)
    ctx = ctx.bound()
    response = ctx.gui.read_internal("operation.cancel", {}, operation_handle=op)
    return {
        "execution": None,
        "op": op,
        "status": response["status"],
        "cancel_requested": True,
    }


def finish_early(ctx: MeasureToolContext, arguments: dict[str, Any]) -> ToolReply:
    """Stop only a recipe's Run, keeping usable partial results for its pipeline."""
    if ("op" in arguments) == ("execution" in arguments):
        raise ValueError("provide exactly one of op or execution")
    if "execution" in arguments:
        recipe = ctx.session.recipes.get(_execution_id(arguments))
    else:
        op = _operation_id(arguments)
        recipe = ctx.session.recipes.for_op(op)
        if recipe is None:
            raise GuiRpcError(
                f"No registered recipe for operation {op}", reason="unknown_operation"
            )
    reply = recipe.finish_early()
    return ToolReply(project_control(reply.data), is_error=reply.is_error)


def build_operation_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "status": {
            "handler": partial(status, ctx),
            "description": "Read execution summary or explicit full detail; global status indexes active executions. Summary previews.run/primary/post are full path lists for session-only PNGs, not persistent saved artifacts. Queries do not refresh guards or attach images.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "execution": {"type": "string", "minLength": 1},
                    "detail": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "default": "summary",
                    },
                },
                "anyOf": [
                    {"required": ["execution"]},
                    {"properties": {"detail": {"enum": ["summary"]}}},
                ],
            },
        },
        "wait": {
            "handler": partial(_wait_tool, ctx),
            "description": "Wait for one operation or execution; timeout does not cancel. Execution summaries include previews.run/primary/post as full session-only PNG path lists. Confirmed persistent outputs are in artifacts; use status(execution, detail=full) for captured native detail.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op": {"type": "integer"},
                    "execution": {"type": "string", "minLength": 1},
                    "timeout": {
                        "type": "number",
                        "default": 60,
                        "minimum": 0,
                        "maximum": 300,
                    },
                },
                "oneOf": [
                    {"required": ["op"], "not": {"required": ["execution"]}},
                    {"required": ["execution"], "not": {"required": ["op"]}},
                ],
            },
        },
        "finish_early": {
            "handler": partial(finish_early, ctx),
            "description": "Stop a recipe Run early and continue with usable partial data.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op": {"type": "integer"},
                    "execution": {"type": "string", "minLength": 1},
                },
                "oneOf": [
                    {"required": ["op"], "not": {"required": ["execution"]}},
                    {"required": ["execution"], "not": {"required": ["op"]}},
                ],
            },
        },
        "cancel": {
            "handler": partial(cancel, ctx),
            "description": "Request cancellation of one operation or execution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op": {"type": "integer"},
                    "execution": {"type": "string", "minLength": 1},
                },
                "oneOf": [
                    {"required": ["op"], "not": {"required": ["execution"]}},
                    {"required": ["execution"], "not": {"required": ["op"]}},
                ],
            },
        },
    }
