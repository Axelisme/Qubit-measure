"""Fixed measure MCP connection and graceful shutdown tools."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.measure.session import GuiRpcError
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


def tool_shutdown(ctx: MeasureToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        result = ctx.send_gui_rpc(
            "app.shutdown", {"discard_unsaved": arguments.get("discard_unsaved", False)}
        )
    except GuiRpcError as exc:
        if exc.code == "timeout":
            return {"stopped": False}
        raise
    pid = result["pid"]
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise RuntimeError("GUI shutdown reply contains an invalid process identity")
    return {"stopped": ctx.bridge.wait_for_gui_exit(pid, timeout=5.0)}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    return {
        "connect": {**CONNECT_TOOL, "handler": partial(tool_connect, ctx)},
        "shutdown": {
            "handler": partial(tool_shutdown, ctx),
            "description": "Close the connected GUI normally. Active operations return busy; "
            "unsaved artifacts require explicit discard_unsaved=true. Wait up to five "
            "seconds for the responding GUI process. A timeout returns stopped=false "
            "without force termination or retry.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "discard_unsaved": {"type": "boolean", "default": False}
                },
            },
        },
    }
