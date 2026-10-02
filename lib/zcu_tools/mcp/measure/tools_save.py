"""Artifact saving through one GUI-owned operation."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.tools_operation import wait


def tab_save(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    """Report reserved destinations only after the save operation succeeds."""
    params = {"tab_id": arguments["tab"]}
    for name in ("artifacts", "paths", "comment"):
        if name in arguments:
            params[name] = arguments[name]
    ctx = ctx.bound()
    submission = ctx.send_gui_rpc("tab.save_artifacts", params)
    op = submission["handle"]
    outcome = wait(ctx, {"op": op, "timeout": 2})
    if outcome["status"] == "finished":
        return {"saved": submission["destinations"]}
    if outcome["status"] == "running":
        return {"op": op}
    error = outcome.get("error") or {}
    raise GuiRpcError(
        f"save operation {op} {outcome['status']}: "
        f"{error.get('message', outcome['status'])}; "
        "read tab_get artifacts for any completed saves",
        reason=f"operation_{outcome['status']}",
        code="precondition_failed",
    )


def build_save_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    keys = {"type": "string", "enum": ["data", "analysis", "post"]}
    return {
        "tab_save": {
            "handler": partial(tab_save, ctx),
            "description": (
                "Save selected artifacts through the GUI's shared drafts. Omitted "
                "artifacts means all saveable artifacts, in analysis/post/data order. "
                "Explicit paths/comment update the drafts; omitted values keep them. "
                "Read tab_get summary/artifacts before saving. No hidden read repairs "
                "stale observations. Return saved actual paths only on success, or op "
                "after a short wait. Use wait and tab_get artifacts.last_saved_path "
                "for long saves and partial failures. Saving cannot be cancelled."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tab": {"type": "string", "minLength": 1},
                    "artifacts": {
                        "oneOf": [
                            {"const": "all"},
                            {
                                "type": "array",
                                "items": keys,
                                "minItems": 1,
                                "uniqueItems": True,
                            },
                        ]
                    },
                    "paths": {
                        "type": "object",
                        "properties": {
                            key: {"type": "string", "minLength": 1}
                            for key in ("data", "analysis", "post")
                        },
                        "additionalProperties": False,
                    },
                    "comment": {"type": "string"},
                },
                "required": ["tab"],
            },
        }
    }
