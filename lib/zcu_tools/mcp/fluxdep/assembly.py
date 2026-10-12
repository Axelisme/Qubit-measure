"""Handwritten Fluxdep agent tools composed over native GUI RPCs."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import (
    StdioLoopHooks,
    ToolTable,
    assemble_tools,
    run_stdio_loop,
)

from .align_tools import build_align_tools
from .fit_tools import build_fit_tools
from .lifecycle import build_control_lifecycle_tools
from .sample_tools import build_sample_tools
from .spectrum_tools import build_spectrum_tools


@dataclass(frozen=True)
class FluxdepServer:
    """A bridge, explicit agent tool table and in-process stdio entry.

    bridge owns its connection and optional launched process. tools composes
    native GUI methods at agent decision points. main disconnects on stdin close
    without shutting down or killing the GUI.
    """

    bridge: McpBridge
    tools: ToolTable
    main: Callable[[], None]


def _control_tools(bridge: McpBridge) -> ToolTable:
    def status(_arguments: dict[str, object]):
        return call_gui(bridge, GuiRpcCall("control.status", {}, 6.0)).data

    def acquire(_arguments: dict[str, object]):
        return call_gui(bridge, GuiRpcCall("control.acquire", {}, 6.0)).data

    def release(_arguments: dict[str, object]):
        return call_gui(bridge, GuiRpcCall("control.release", {}, 6.0)).data

    prefix = bridge.config.tool_prefix
    return {
        prefix + "control_status": {
            "handler": status,
            "description": "Read the native GUI control identity.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        prefix + "control_acquire": {
            "handler": acquire,
            "description": "Request GUI control for this fixed owner connection.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        prefix + "control_release": {
            "handler": release,
            "description": "Release GUI control to Local without stopping accepted work.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    }


def build_fluxdep_server(
    config: MCPBridgeConfig,
    repo_root: Path,
    *,
    bridge: McpBridge | None = None,
) -> FluxdepServer:
    """Assemble explicit tools and lifecycle callbacks over one GUI bridge.

    config supplies transport settings/instructions; repo_root anchors GUI launch.
    An injected bridge must use this config or ValueError is raised. GUI owns
    business validation, identity and operation state; MCP only composes calls,
    summarizes native facts and requests evidence. Invocation/image errors
    propagate to stdio tool-error content, without retry or rollback. No business
    events are subscribed, and cleanup never stops the GUI.
    """
    if bridge is not None and bridge.config != config:
        raise ValueError("Injected bridge must use the supplied Fluxdep config")
    bridge = bridge if bridge is not None else McpBridge(config)
    lifecycle, names = build_control_lifecycle_tools(config, bridge, repo_root)
    tools = assemble_tools(
        build_spectrum_tools(bridge)
        | build_align_tools(bridge)
        | build_sample_tools(bridge)
        | build_fit_tools(bridge)
        | _control_tools(bridge),
        lifecycle,
        names,
    )

    def cleanup() -> None:
        bridge.disconnect()

    def main() -> None:
        run_stdio_loop(config, tools, hooks=StdioLoopHooks(on_cleanup=cleanup))

    return FluxdepServer(bridge, tools, main)
