"""Fixed cfg tools that delegate editing to the GUI-owned draft."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def tab_edit(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Apply ordered edits to the tab's live draft using agent sweep grammar."""
    return ctx.send_gui_rpc(
        "tab.set_cfg",
        {"tab_id": arguments["tab"], "edits": arguments["edits"], "agent_edit": True},
    )


CFG_TOOLS: dict[str, dict[str, Any]] = {
    "tab_edit": {
        "handler": tab_edit,
        "description": "Edit an explicit tab's live cfg draft in order using canonical paths and agent whole-sweep objects. Stop on the first error without rolling back earlier edits; the error names the failed path and applied prefix. Returns {applied, valid, actual, removed, added}, with normalized sweep values in actual. Cannot edit a running tab.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab": {"type": "string", "minLength": 1},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "minLength": 1},
                            "value": {},
                        },
                        "required": ["path", "value"],
                    },
                },
            },
            "required": ["tab", "edits"],
        },
    },
}


def build_cfg_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in CFG_TOOLS.items()
    }
