"""Fixed measure MCP connection tool."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.tool_context import MeasureToolContext
from zcu_tools.mcp.measure.tools_operation import status


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
    token = arguments.get("token")
    if token is not None and (not isinstance(token, str) or not token):
        raise ValueError("token must be a non-empty string")
    connection = ctx.session.connect_to_gui(
        port=port, launch=launch, clean=clean, token=token
    )
    return {**connection, "status": status(ctx, {})}


CONNECT_TOOL: dict[str, Any] = {
    "handler": tool_connect,
    "description": "Attach to the live GUI or launch one when requested. Supply token if the GUI requires authentication; reconnect reuses it. Never connects hardware; authentication failures and incompatible wire contracts are distinct errors.",
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
            "token": {
                "type": "string",
                "minLength": 1,
                "description": "GUI control token, if configured. Treat as a secret.",
            },
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {"connect": {**CONNECT_TOOL, "handler": partial(tool_connect, ctx)}}
