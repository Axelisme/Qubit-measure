"""Fixed measure MCP connection tool."""

from __future__ import annotations

from functools import partial
from typing import Any

from zcu_tools.mcp.core.control import ControlConnection
from zcu_tools.mcp.measure.session import ConnectIdentity, GuiStartupPaths
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
    # Validate supplied values before network work; omitted values retain identity.
    ControlConnection.parse(
        arguments.get("label", "agent"), arguments.get("mode", "owner")
    )
    simulate = arguments.get("simulate")
    if simulate is not None and (not isinstance(simulate, str) or not simulate):
        raise ValueError("simulate must be a non-empty environment Python path")
    project_root = arguments.get("project_root")
    if project_root is not None and not isinstance(project_root, str):
        raise ValueError("project_root must be a string")
    connection = ctx.session.connect_to_gui(
        port=port,
        startup_paths=GuiStartupPaths(simulate=simulate, project_root=project_root),
        launch=launch,
        clean=clean,
        token=token,
        identity=ConnectIdentity(arguments.get("label"), arguments.get("mode")),
    )
    return {**connection, "status": status(ctx, {})}


CONNECT_TOOL: dict[str, Any] = {
    "handler": tool_connect,
    "description": "Attach to the live GUI or launch one when requested. Initially label=agent and mode=owner; readonly never acquires. Reconnect retains omitted identity options; changing them reopens the socket. Return native control status even when the GUI is locked or another holder controls it. Supply token if required. Never connects hardware; authentication and incompatible wire contracts are distinct errors.",
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
            "simulate": {
                "type": "string",
                "minLength": 1,
                "description": "Trusted environment Python path; only applies when launching a new GUI. Attach with a path is rejected.",
            },
            "project_root": {
                "type": "string",
                "description": "Data project root for a new GUI. Must be an absolute path to an existing directory; the launcher validates it. Attach with a root is rejected. Omit to use the repo root.",
            },
            "label": {"type": "string", "minLength": 1, "default": "agent"},
            "mode": {
                "type": "string",
                "enum": ["owner", "readonly"],
                "default": "owner",
            },
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
