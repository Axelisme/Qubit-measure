"""Fixed experiment run and analysis tools over GUI-owned operations."""

from __future__ import annotations

import base64
from functools import partial
from pathlib import Path
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.tools_operation import wait
from zcu_tools.mcp.measure.tools_tab import tab_get


def tab_run(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, int]:
    """Start exactly the supplied cfg ref once; completion belongs to wait()."""
    tab = arguments.get("tab")
    if not isinstance(tab, str) or not tab:
        raise ValueError("tab must be a non-empty string")
    reply = ctx.send_gui_rpc(
        "tab.run_start", {"tab_id": tab, "expected": arguments["expected"]}
    )
    return {"op": reply["handle"]}


def tab_analyze(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
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
    started = ctx.send_gui_rpc(method, {"tab_id": tab, "updates": params})
    op = started["handle"]
    if started["interactive"]:
        return {"status": "interactive", "op": op}
    outcome = wait(ctx, {"op": op, "timeout": 2.0})
    if outcome["status"] != "finished":
        return {**outcome, "op": op}
    section = "analysis" if stage == "primary" else "post"
    result = tab_get(ctx, {"tab": tab, "include": [section]})[section]
    if "summary" not in result:
        raise RuntimeError("finished analysis has no result")
    return {
        "status": "finished",
        "summary": result["summary"],
        "figure": result["figure"],
        "params": started["params"],
        "invalidated": started["invalidated_on_success"],
    }


def tab_interact(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
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
    reply = dict(ctx.send_gui_rpc("tab.interact", params))
    figure = reply["figure"]
    if figure is not None:
        png = base64.b64decode(figure["png_b64"], validate=True)
        path = ctx.session.new_png_path()
        Path(path).write_bytes(png)
        reply["figure"] = path
    return reply


def build_run_analyze_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "tab_interact": {
            "handler": partial(tab_interact, ctx),
            "description": "Read the active interactive plugin's committed state, commands, info and figure without changing focus. Supply payload={command,args} to execute one command and follow the Analysis pane. done settles the original operation; cancel uses cancel(op). Best-effort last commit wins; no seen guard, hidden reads or retry. Figure is a session-owned PNG path or null; preview_active distinguishes local preview from committed state.",
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
            "description": "Analyze the GUI tab with shared parameters. Wait briefly for summary, figure, effective params and invalidated content; otherwise return an operation. Interactive analysis returns immediately. No hidden reads or retry.",
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
        "tab_run": {
            "handler": partial(tab_run, ctx),
            "description": "Start a run with the explicitly observed cfg ref {cfg_id,revision} and return a waitable operation. revision is a canonical decimal string. Stale or non-Valid cfg is rejected; no hidden reads, refresh or retry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string"},
                    "expected": {"type": "object"},
                },
                "required": ["tab", "expected"],
            },
        },
    }
