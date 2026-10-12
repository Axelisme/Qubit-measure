"""Handwritten fit decision tools over authoritative GUI RPCs."""

from typing import TypedDict

from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import ToolTable


class DatabaseArguments(TypedDict):
    """path names a GUI-local search database to inspect, without loading a fit."""

    path: str


def build_fit_tools(bridge: McpBridge) -> ToolTable:
    """Build native database inspection tools without retaining fit state.

    GUI validates paths and owns I/O failures. Invocation errors propagate
    without retry, reconnect or rollback.
    """

    def database_info(arguments: DatabaseArguments) -> ToolReply:
        return call_gui(bridge, GuiRpcCall("fit.database_info", dict(arguments), 31.0))

    return {
        bridge.config.tool_prefix + "database_info": {
            "handler": database_info,
            "description": "Inspect a search database's parameter/flux ranges, entry count and energy levels without running search.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    }
