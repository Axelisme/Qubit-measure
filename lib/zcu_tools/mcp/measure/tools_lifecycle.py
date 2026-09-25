"""The fixed measure MCP connection tool."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.tools_overview import assemble_overview


def tool_connect(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    port = arguments.get("port")
    if port is not None and (isinstance(port, bool) or not isinstance(port, int)):
        raise ValueError("port must be an integer")
    launch = arguments.get("launch", "never")
    if launch not in ("never", "if_missing", "new"):
        raise ValueError("launch must be never, if_missing or new")
    clean = arguments.get("clean", False)
    if not isinstance(clean, bool):
        raise ValueError("clean must be a boolean")
    connection = ctx.session.connect_to_gui(port=port, launch=launch, clean=clean)
    return {**connection, "status": assemble_overview(ctx)}


CONNECT_TOOL: dict[str, Any] = {
    "handler": tool_connect,
    "description": "Attach to the live GUI or launch one when requested. Never connects hardware; incompatible wire contracts fail before catalog load.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "port": {"type": "integer"},
            "launch": {
                "type": "string",
                "enum": ["never", "if_missing", "new"],
                "default": "never",
            },
            "clean": {"type": "boolean", "default": False},
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {"connect": {**CONNECT_TOOL, "handler": partial(tool_connect, ctx)}}
