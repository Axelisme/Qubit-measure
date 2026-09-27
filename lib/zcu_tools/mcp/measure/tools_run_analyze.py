"""Fixed experiment run and analysis tools over GUI-owned operations."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def tab_run(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, int]:
    """Start the attached tab's current GUI draft; completion belongs to wait()."""
    tab = arguments.get("tab")
    if not isinstance(tab, str) or not tab:
        raise ValueError("tab must be a non-empty string")
    reply = ctx.send_gui_rpc("tab.run_start", {"tab_id": tab})
    return {"op": reply["handle"]}


def build_run_analyze_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "tab_run": {
            "handler": partial(tab_run, ctx),
            "description": "Start a run with the GUI tab's current cfg and return a waitable operation.",
            "inputSchema": {
                "type": "object",
                "properties": {"tab": {"type": "string"}},
                "required": ["tab"],
            },
        },
    }
