"""Fixed analysis tools over GUI-owned operations."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.execution_reply import project_execution
from zcu_tools.mcp.measure.interaction import handoff_interaction, interact
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

logger = logging.getLogger(__name__)


def tab_analyze(ctx: MeasureToolContext, arguments: dict[str, Any]) -> ToolReply:
    """Start once; expose completion only after the GUI operation settles."""
    tab = arguments.get("tab")
    stage = arguments.get("stage", "primary")
    params = arguments.get("params", {})
    if not isinstance(tab, str) or not tab:
        raise ValueError("tab must be a non-empty string")
    if stage not in ("primary", "post"):
        raise ValueError("stage must be primary or post")
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    ctx = ctx.bound()
    execution = ctx.session.executions.start(ctx.gui, tab, stage)
    try:
        started = ctx.gui.send_gui_rpc(
            method,
            {"tab_id": tab, "updates": params},
            before_send=execution.admit_start,
        )
        ctx.session.executions.accept_start(execution, started)
    except Exception as exc:
        execution.fail_start(exc)
        logger.exception("Analysis start interrupted; execution receipt retained")
    else:
        if started["interactive"]:
            handoff_interaction(ctx, execution)
    reply = execution.wait(0 if execution.snapshot().status == "interactive" else 2.0)
    return ToolReply(
        project_execution(reply.data),
        reply.images,
        reply.is_error,
    )


def tab_interact(ctx: MeasureToolContext, arguments: dict[str, Any]) -> ToolReply:
    """Forward one plugin command or read; retain the GUI's committed state."""
    tab = arguments.get("tab")
    if not isinstance(tab, str) or not tab:
        raise ValueError("tab must be a non-empty string")
    params: dict[str, Any] = {"tab_id": tab}
    if "payload" in arguments:
        payload = arguments["payload"]
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        params["payload"] = payload
    reply = interact(ctx.bound(), params)
    if params.get("payload", {}).get("command") == "done":
        return ToolReply(
            project_execution(reply.data),
            reply.images,
            reply.is_error,
        )
    return reply


def build_run_analyze_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "tab_interact": {
            "handler": partial(tab_interact, ctx),
            "description": "Read the active interactive plugin's committed state, commands, info and figure without changing focus. Supply payload={command,args} to execute one command and follow the Analysis pane. done settles the original operation and returns the shared execution summary; cancel uses cancel(op). Best-effort last commit wins; no seen guard, hidden reads or retry. Available figure bytes attach as in-memory MCP image content, without a file path; preview_active distinguishes local preview from committed state. Later tab.get_figure reads the live GUI image, not a saved capture.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["tab"],
            },
        },
        "tab_analyze": {
            "handler": partial(tab_analyze, ctx),
            "description": "Analyze the GUI tab with shared parameters. Wait briefly for an execution summary with effective params, per-pane analysis/save steps and in-memory MCP image content. Complete synchronous Save reports are in artifacts, including JSON and all PNG outputs. Use status(execution, detail=full) for captured native detail; otherwise wait by execution. Interactive analysis immediately reads and returns the active state, commands and available image with tab/op. Post result-read failures retain source-bound recovery RPC/params and original op/stage. First compare tab.snapshot post_analysis_state.source_operation_id with recovery.params.operation_id; if different, report superseded. Otherwise rpc_call the supplied route; it rejects replacement races. Never rerun analysis to recover a read. Post read budgets come from the owner (5 seconds plus 5 per tone, maximum 60; absent=5). No hidden pre-reads or retry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string"},
                    "stage": {
                        "type": "string",
                        "enum": ["primary", "post"],
                        "default": "primary",
                    },
                    "params": {"type": "object"},
                },
                "required": ["tab"],
            },
        },
    }
