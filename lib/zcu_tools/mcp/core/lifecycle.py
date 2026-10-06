"""Shared explicit GUI launch, connect and disconnect tools."""

from pathlib import Path
from typing import Any

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig, resolve_connect_port
from zcu_tools.mcp.core.stdio_server import Tool, ToolTable


def build_lifecycle_tools(
    config: MCPBridgeConfig, bridge: McpBridge, repo_root: Path, gui_name: str
) -> tuple[ToolTable, frozenset[str]]:
    """Build the three hand-written lifecycle tools (connect/disconnect/launch).

    ``gui_name`` is the human GUI name used in the tool prose (e.g. ``fluxdep-gui``,
    ``dispersive-fit-gui``) — distinct from ``config.server_display_name`` (the MCP
    ``serverInfo`` name).

    The tools never stop the GUI. Cleanup policy belongs to the assembling app.
    The result contains the tool table and its names for collision-safe assembly.
    """
    prefix = config.tool_prefix
    port = config.default_port

    def _connect(arguments: dict[str, Any]) -> str:
        requested = arguments.get("port")
        if requested is not None and not isinstance(requested, int):
            raise ValueError("Invalid 'port' argument (must be integer)")
        p = resolve_connect_port(config, requested)
        return bridge.connect(p, arguments.get("token"))

    def _disconnect(arguments: dict[str, Any]) -> str:
        del arguments
        return bridge.disconnect()

    def _launch(arguments: dict[str, Any]) -> str:
        p = int(arguments.get("port", port))
        token: str | None = arguments.get("token")
        auto_connect = bool(arguments.get("auto_connect", True))
        return bridge.launch(repo_root, p, token, auto_connect=auto_connect)

    overrides: dict[str, Tool] = {
        f"{prefix}connect": {
            "handler": _connect,
            "description": (
                f"Connect the MCP bridge to an ALREADY-RUNNING {gui_name}'s TCP "
                f"control port. Omit 'port' to auto-discover the running GUI (reads "
                f"the session file the GUI writes; covers the case where it fell "
                f"back off port {port}), falling back to port {port} if none is "
                f"found. Errors if no GUI is listening — use {prefix}launch to start "
                f"one. Skip this if you used {prefix}launch with auto_connect=true "
                f"(default)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "port": {
                        "type": "integer",
                        "description": (
                            f"TCP port of a running GUI control service. Omit to "
                            f"auto-discover (then fall back to {port})."
                        ),
                    },
                    "token": {
                        "type": "string",
                        "description": "Optional authentication token",
                    },
                },
            },
        },
        f"{prefix}disconnect": {
            "handler": _disconnect,
            "description": (
                "Disconnect the MCP bridge from the GUI control port. Does NOT "
                "stop the GUI process — it keeps running for the user to drive."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        f"{prefix}launch": {
            "handler": _launch,
            "description": (
                f"Launch the {gui_name} as a NEW subprocess on a TCP control "
                f"port (default {port}), wait until ready, and optionally connect. "
                f"Use as the first step. Errors if the port is already in use (a "
                f"stale GUI). By default auto_connect=true."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "port": {
                        "type": "integer",
                        "description": (
                            f"TCP control port for the GUI (default {port})"
                        ),
                    },
                    "token": {
                        "type": "string",
                        "description": "Optional shared auth token",
                    },
                    "auto_connect": {
                        "type": "boolean",
                        "default": True,
                        "description": (
                            f"Call {prefix}connect automatically once ready "
                            "(default true)"
                        ),
                    },
                },
            },
        },
    }
    names = frozenset(overrides)
    return overrides, names
