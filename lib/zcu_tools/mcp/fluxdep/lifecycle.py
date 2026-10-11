"""Fluxdep lifecycle tools initialize fixed identity only on new connections."""

from pathlib import Path
from typing import Any

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.control import ControlConnection, probe_control_wire
from zcu_tools.mcp.core.lifecycle import build_lifecycle_tools
from zcu_tools.mcp.core.stdio_server import ToolTable


def build_control_lifecycle_tools(
    config: MCPBridgeConfig, bridge: McpBridge, repo_root: Path
) -> tuple[ToolTable, frozenset[str]]:
    """Attach identity handshake to Fluxdep connect and auto-connected launch.

    label/mode default to agent/owner initially and retain the selected values on
    reconnect. Auth and wire compatibility precede initialization. Return native
    control and versions with the lifecycle note. Failed handshakes disconnect;
    GUI lock/other holder is a successful connect without control. Dispersive's
    shared readonly lifecycle remains unchanged.
    """
    tools, names = build_lifecycle_tools(config, bridge, repo_root, "fluxdep-gui")
    identity = ControlConnection()

    def initialize(note: str, selected: ControlConnection) -> dict[str, object]:
        nonlocal identity
        try:
            versions = probe_control_wire(bridge)
            control = selected.initialize(bridge)
        except Exception:  # A failed handshake must not leave a writable socket.
            bridge.disconnect()
            raise
        identity = selected
        return {"connection": note, "versions": versions, "control": control}

    for suffix in ("connect", "launch"):
        name = config.tool_prefix + suffix
        native_handler = tools[name]["handler"]

        def connect(
            arguments: dict[str, Any],
            *,
            handler=native_handler,
            is_launch=suffix == "launch",
        ) -> str | dict[str, object]:
            selected = ControlConnection(
                arguments.get("label", identity.label),
                arguments.get("mode", identity.mode),
            )
            note = handler(arguments)
            if is_launch and not arguments.get("auto_connect", True):
                return note
            return initialize(note, selected)

        tools[name]["handler"] = connect
        tools[name]["inputSchema"]["properties"].update(
            {
                "label": {"type": "string", "minLength": 1, "default": "agent"},
                "mode": {
                    "type": "string",
                    "enum": ["owner", "readonly"],
                    "default": "owner",
                },
            }
        )
    return tools, names
