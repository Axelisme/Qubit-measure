"""Fixed cfg tools that delegate atomic editing to the GUI-owned resource."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def tab_edit(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    return ctx.send_gui_rpc(
        "tab.edit_cfg",
        {
            "tab_id": arguments["tab"],
            "expected": arguments["expected"],
            "edits": arguments["edits"],
        },
    )


CFG_TOOLS: dict[str, dict[str, Any]] = {
    "tab_edit": {
        "handler": tab_edit,
        "description": (
            "Atomically edit an explicit tab's cfg at the cfg_ref returned by a complete "
            "tab_get(include=['cfg']) observation. Pass that cfg_ref as expected. Paths are arrays "
            "of field-name segments. Input tags are __expr, __text, __complex and __ref; "
            "sweeps are complete range objects. Stale versions and malformed batches are "
            "rejected without publishing a prefix. A successful edit can publish Invalid "
            "while input is incomplete. Returns the new resource observation. No hidden "
            "read or automatic retry; busy tabs reject manual edits."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tab": {"type": "string", "minLength": 1},
                "expected": {
                    "type": "object",
                    "properties": {
                        "cfg_id": {"type": "string", "minLength": 1},
                        "revision": {"type": "string", "pattern": "^(0|[1-9][0-9]*)$"},
                    },
                    "required": ["cfg_id", "revision"],
                    "additionalProperties": False,
                },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                            "value": {},
                        },
                        "required": ["path", "value"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["tab", "expected", "edits"],
            "additionalProperties": False,
        },
    },
}


def build_cfg_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        name: {**entry, "handler": partial(entry["handler"], ctx)}
        for name, entry in CFG_TOOLS.items()
    }
