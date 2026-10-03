"""Fixed tab closing tool over the GUI owner."""

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def tab_close(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    tab = arguments["tab"]
    ctx.send_gui_rpc(
        "tab.close",
        {
            "tab_id": tab,
            "discard_unsaved": arguments.get("discard_unsaved", False),
        },
    )
    return {"closed": tab}


def build_tab_read_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "tab_close": {
            "handler": partial(tab_close, ctx),
            "description": "Close the specified idle tab. All unsaved artifacts require "
            "discard_unsaved=true. Active operations cannot be discarded. No hidden reads or retry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Explicit GUI tab id",
                    },
                    "discard_unsaved": {"type": "boolean", "default": False},
                },
                "required": ["tab"],
            },
        },
    }
