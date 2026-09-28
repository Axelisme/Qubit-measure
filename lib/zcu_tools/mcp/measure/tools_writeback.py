"""Agent writeback view over the GUI-owned shared draft."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext


def writeback(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    stage = arguments.get("stage", "primary")
    if stage not in ("primary", "post"):
        raise ValueError("stage must be primary or post")
    params = {
        "tab_id": arguments["tab"],
        "subtab_id": "analysis" if stage == "primary" else "post_analysis",
    }
    if "write" in arguments:
        return ctx.send_gui_rpc(
            "tab.writeback_write", {**params, "write": arguments["write"]}
        )
    preview = ctx.send_gui_rpc("tab.writeback_preview", params)
    return {
        "destination": preview["destination_context"],
        "items": [
            {
                "id": item["id"],
                "kind": "md" if item["kind"] == "metadict" else item["kind"],
                "target": item["target_name"],
                "description": item["description"],
                "current": item["current"],
                "proposed": item["proposed"],
            }
            for item in preview["items"]
        ],
    }


WRITEBACK_TOOL: dict[str, Any] = {
    "handler": writeback,
    "description": (
        "Preview shared GUI writeback proposals and their active-context destination. "
        "With write=[{id,target?,value?,edits?}], edit the listed draft items in order, "
        "then apply only those IDs once. A draft edit failure preserves the successful "
        "prefix and starts no context write. GUI checkboxes do not change. "
        "Returns {written:[{id,kind,target,before,after}]} with actual complete values; "
        "same target names across kinds remain separate. No cross-file rollback. "
        "Omitted value retains the proposal; explicit null sets an md value to null. "
        "Read the tab and context explicitly before a guarded write."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "tab": {"type": "string", "minLength": 1},
            "stage": {"type": "string", "enum": ["primary", "post"]},
            "write": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "target": {"type": "string", "minLength": 1},
                        "value": {},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string", "minLength": 1},
                                    "value": {},
                                },
                                "required": ["path", "value"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["tab"],
        "additionalProperties": False,
    },
}


def build_writeback_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {"writeback": {**WRITEBACK_TOOL, "handler": partial(writeback, ctx)}}
