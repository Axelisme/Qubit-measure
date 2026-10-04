"""Fixed analysis tools over GUI-owned operations."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure.execution_reply import project_execution
from zcu_tools.mcp.measure.interaction import handoff_interaction, interact
from zcu_tools.mcp.measure.tool_context import MeasureToolContext


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
    started = ctx.send_gui_rpc(method, {"tab_id": tab, "updates": params})
    execution = ctx.session.executions.start(ctx.gui, tab, stage, started)
    if started["interactive"]:
        handoff_interaction(ctx, execution)
    reply = execution.wait(0 if started["interactive"] else 2.0)
    return ToolReply(project_execution(reply.data), reply.images, reply.is_error)


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
        return ToolReply(project_execution(reply.data), reply.images, reply.is_error)
    return reply


def build_run_analyze_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "tab_interact": {
            "handler": partial(tab_interact, ctx),
            "description": "Read the active interactive plugin's committed state, commands, info and figure without changing focus. Supply payload={command,args} to execute one command and follow the Analysis pane. done settles the original operation; cancel uses cancel(op). Best-effort last commit wins; no seen guard, hidden reads or retry. Figure is a session-owned absolute PNG path or null, accompanied by MCP image content when available; preview_active distinguishes local preview from committed state.",
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
            "description": "Analyze the GUI tab with shared parameters. Wait briefly for summary, figure path with MCP image content, effective params and invalidated content; otherwise return an operation. Interactive analysis immediately reads and returns the active state, commands and available image with tab/op. No hidden pre-reads or retry.",
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
