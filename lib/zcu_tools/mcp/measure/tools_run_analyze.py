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
            "description": "Interact with the active analysis plugin. Omit payload to read committed state, commands, info and image without changing focus. Commands follow Analysis; done joins original completion and recipe continuation. preview_active marks local preview. Last commit wins; no retry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string"},
                    "payload": {
                        "type": "object",
                        "description": "One plugin action: {command, args}. Use commands from the committed state.",
                    },
                },
                "required": ["tab"],
            },
        },
        "tab_analyze": {
            "handler": partial(tab_analyze, ctx),
            "description": "Analyze existing tab data, save analysis artifacts and return an execution summary/images. Interactive analysis returns state, commands and tab/op. presentation records source run_id and replaced pane figures, not file overwrites. Post result-read failures retain source-bound recovery; follow the run-measure-gui recovery procedure, not reanalysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string"},
                    "stage": {
                        "type": "string",
                        "enum": ["primary", "post"],
                        "default": "primary",
                    },
                    "params": {
                        "type": "object",
                        "description": "Named analysis parameter updates in native units. Omit to retain current values.",
                    },
                },
                "required": ["tab"],
            },
        },
    }
