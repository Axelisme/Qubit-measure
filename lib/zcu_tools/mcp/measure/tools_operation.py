"""Fixed operation tools over the GUI-owned operation and session state."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from functools import partial
from typing import Any

from zcu_tools.experiment.scope import parse_cancel_scope
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.execution_reply import (
    ReplyDetail,
    project_control,
    project_execution,
)
from zcu_tools.mcp.measure.script_recipe_execution import (
    ScriptRecipeExecution,
    ScriptRecipeSnapshot,
)
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.tools_workflow import cancel_workflow


def _execution_status(
    ctx: MeasureToolContext, key: str, detail: ReplyDetail
) -> dict[str, Any]:
    """Project a local recipe or standalone execution without reconnecting."""
    if key.startswith("recipe-"):
        execution = ctx.session.recipes.get(key)
        snapshot = asdict(execution.snapshot())
        paths = asdict(execution.scope_status())
        return {
            **project_execution(snapshot, detail=detail),
            **paths,
        }
    return project_execution(
        asdict(ctx.session.executions.get(key).snapshot()),
        detail=detail,
    )


def _recent_terminal_recipes(
    snapshots: tuple[ScriptRecipeSnapshot, ...],
) -> list[ScriptRecipeSnapshot]:
    return sorted(
        (item for item in snapshots if item.finished_at is not None),
        key=lambda item: item.finished_at or "",
        reverse=True,
    )[:10]


def status(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Read one execution, or GUI orientation with installed point file versions.

    With no execution argument, files mirrors entry.info: ordered parameters,
    library and aliases records with absolute paths and installed SHA-256
    revisions. Missing aliases uses null; no selected point returns []. This
    query does not read point files or implicitly refresh them.
    """
    detail = arguments.get("detail", "summary")
    if detail not in ("summary", "full"):
        raise ValueError("detail must be summary or full")
    if detail == "full" and "execution" not in arguments:
        raise ValueError("detail=full requires execution")
    if "execution" in arguments:
        return _execution_status(ctx, _execution_id(arguments), detail)
    session = ctx.gui
    has_project = bool(session.read_internal("state.has_project", {})["value"])
    has_context = bool(session.read_internal("state.has_active_context", {})["value"])
    has_soc = bool(session.read_internal("state.has_soc", {})["value"])

    entry_info = (
        session.read_internal("entry.info", {})
        if has_project
        else {"entry": None, "files": []}
    )
    entry = entry_info["entry"]
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
    missing = [
        name
        for name, present in (
            ("entry", has_project),
            ("active_context", has_context),
            ("soc", has_soc),
            ("tab", bool(tabs)),
        )
        if not present
    ]
    recipe_snapshots = ctx.session.recipes.snapshots()
    execution_snapshots = [
        asdict(item) for item in ctx.session.executions.snapshots()
    ] + [
        {
            **asdict(item),
            **asdict(ctx.session.recipes.get(item.execution).scope_status()),
        }
        for item in recipe_snapshots
    ]
    return {
        "entry": entry,
        "files": entry_info["files"],
        "soc": soc,
        "context": {"active": session.read_internal("context.active", {})["label"]},
        "devices": [
            {**device, "connected": device["status"] == "connected"}
            for device in session.read_internal("device.list", {})["devices"]
        ],
        "ready": {"can_run": not missing, "missing": missing},
        "tabs": tabs,
        "executions": [
            {
                **project_execution(item),
                **(
                    {
                        "active_path": item["active_path"],
                        "live_paths": item["live_paths"],
                    }
                    if "active_path" in item
                    else {}
                ),
            }
            for item in execution_snapshots
            if item["phase"] != "terminal"
        ],
        "recent_terminal_executions": [
            {
                "execution": item.execution,
                "recipe": item.recipe,
                "status": item.status,
                "finished_at": item.finished_at,
            }
            for item in _recent_terminal_recipes(recipe_snapshots)
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
        data = project_execution(reply.data)
        return ToolReply(
            data
            if key.startswith("recipe-")
            else {**data, "elapsed_s": max(0.0, time.monotonic() - start)},
            reply.images,
            # Accepted failures remain outcome data; interactive delivery errors
            # still flag the failed handoff without rewriting the native outcome.
            is_error=reply.is_error and reply.data["status"] == "interactive",
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
    if reply["reason"] != "timeout":
        raise ValueError(f"unexpected operation await reason: {reply['reason']!r}")
    result["status"] = "running"
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
    """Cancel exactly one execution, native workflow run, or GUI operation."""
    if sum(key in arguments for key in ("op", "execution", "run_id")) != 1:
        raise ValueError("provide exactly one of op, execution or run_id")
    scope = arguments.get("scope", "script")
    expected = arguments.get("expected")
    label, sequence = parse_cancel_scope(scope, expected)
    if "run_id" in arguments:
        run_id = arguments["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        return cancel_workflow(
            ctx,
            {
                key: value
                for key, value in arguments.items()
                if key in {"run_id", "scope", "expected"}
            },
        )
    target_op = None
    if "execution" in arguments:
        key = _execution_id(arguments)
        execution = (
            ctx.session.recipes.get(key)
            if key.startswith("recipe-")
            else ctx.session.executions.get(key)
        )
    else:
        op = _operation_id(arguments)
        if label in {"schedule", "acquire"} or "/" in label:
            target_op = op
        execution = ctx.session.recipes.for_op(op) or ctx.session.executions.for_op(op)
    if execution is not None:
        if isinstance(execution, ScriptRecipeExecution):
            reply = execution.cancel(
                scope=scope if "scope" in arguments else "recipe",
                expected=expected,
                op=target_op,
            )
        elif label != "script" or sequence is not None:
            raise ValueError("scoped cancellation requires a recipe or workflow")
        else:
            reply = execution.cancel()
        return ToolReply(project_control(reply.data), is_error=reply.is_error)
    if label != "script" or sequence is not None:
        raise ValueError("scoped cancellation requires a recipe or workflow")
    op = _operation_id(arguments)
    ctx = ctx.bound()
    response = ctx.gui.read_internal("operation.cancel", {}, operation_handle=op)
    return {
        "execution": None,
        "op": op,
        "status": response["status"],
        "cancel_requested": True,
    }


def build_operation_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "status": {
            "handler": partial(status, ctx),
            "description": "Read execution summary or explicit full detail; global status indexes active executions and the 10 most recently finished recipes in recent_terminal_executions (execution, recipe, status, UTC finished_at), newest first. Terminal recipe replies retain execution; failures include failure.phase and failure.reason. Global ready.can_run checks environment readiness (entry, active context, SoC and any tab), not Form validity. Before Run, read the target tab.get_cfg input/values/errors even when ready.can_run is true. Artifacts preserve complete native Save reports. Global files mirrors entry.info with ordered parameters/library/aliases document, absolute path and installed SHA-256 revision; missing aliases uses null, no selected point returns []. For direct-file edits use rpc_call(context.reload) with the installed base revision. Queries do not refresh guards, read files or attach images.",
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
            "description": "Wait for one operation or execution; timeout does not cancel. Terminal recipes retain execution; failed recipes include failure.phase and the human failure.reason without a traceback. Execution replies may attach in-memory image content without session files. Complete native Save outputs are in artifacts; use status(execution, detail=full) for captured native detail.",
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
        "cancel": {
            "handler": partial(cancel, ctx),
            "description": "Omit scope to stop the target root. Select a live kind, kind#N, operation:name or exact full live_paths path; ambiguous matches return full-path candidates. Recipe run requires an active tab.run. Acknowledgement does not imply cleanup completed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op": {"type": "integer"},
                    "execution": {"type": "string", "minLength": 1},
                    "run_id": {"type": "string", "minLength": 1},
                    "scope": {"type": "string"},
                    "expected": {"type": "integer", "minimum": 1},
                },
                "oneOf": [
                    {
                        "required": ["op"],
                        "not": {
                            "anyOf": [
                                {"required": ["execution"]},
                                {"required": ["run_id"]},
                            ]
                        },
                    },
                    {
                        "required": ["execution"],
                        "not": {
                            "anyOf": [{"required": ["op"]}, {"required": ["run_id"]}]
                        },
                    },
                    {
                        "required": ["run_id"],
                        "not": {
                            "anyOf": [{"required": ["op"]}, {"required": ["execution"]}]
                        },
                    },
                ],
            },
        },
    }
