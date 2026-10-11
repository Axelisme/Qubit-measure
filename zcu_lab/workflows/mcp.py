"""T1 workflow MCP attachment over the existing GUI RPC owners."""

import base64
from pathlib import Path

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.images import validated_png
from zcu_tools.mcp.core.lifecycle import build_lifecycle_tools
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.rpc import GuiRpcCall, call_gui
from zcu_tools.mcp.core.stdio_server import (
    StdioLoopHooks,
    ToolTable,
    assemble_tools,
    generate_tools,
    run_stdio_loop,
)

from zcu_lab.workflows.gui.remote_methods import METHOD_SPECS


def build_workflow_tools(
    config: MCPBridgeConfig, bridge: McpBridge, repo_root: Path
) -> ToolTable:
    """Assemble the nine workflow RPC tools and shared lifecycle callbacks.

    config supplies tool names and connection/launch settings; bridge must use
    that same config, otherwise raise ValueError before assembling any tools.
    repo_root anchors the configured GUI launch script. Every workflow tool sends
    exactly one declared RPC through this bridge, without prereads, retries,
    subscriptions, draft state or operation tracking. GUI invocation errors retain
    code/message/reason; transport errors propagate and timeouts do not cancel.
    workflow.figure validates its one returned PNG with the shared image owner,
    removes base64 from text and returns run_id/name plus inline image content.
    Corrupt base64/PNG raises without retry or file creation. All other native
    replies are unchanged, including failed operation outcomes.
    """
    if bridge.config != config:
        raise ValueError("bridge config must match workflow MCP config")

    def _send(
        method: str, params: dict[str, object], *, timeout_seconds: float
    ) -> dict[str, object]:
        return call_gui(bridge, GuiRpcCall(method, params, timeout_seconds)).data

    generated = generate_tools(config, METHOD_SPECS, frozenset(), _send)
    figure_spec = METHOD_SPECS["workflow.figure"]
    figure_name = figure_spec.tool_name or config.tool_prefix + "workflow_figure"
    forward_figure = generated[figure_name]["handler"]

    def _figure(arguments: dict[str, object]) -> ToolReply:
        data: dict[str, object] = forward_figure(arguments)
        encoded = data.pop("png_base64")
        if not isinstance(encoded, str):
            raise ValueError("workflow.figure png_base64 must be a string")
        image = validated_png(base64.b64decode(encoded, validate=True))
        return ToolReply(data, images=(image,))

    generated[figure_name]["handler"] = _figure
    lifecycle, lifecycle_names = build_lifecycle_tools(
        config, bridge, repo_root, "workflows-gui"
    )
    return assemble_tools(generated, lifecycle, lifecycle_names)


def run_workflow_mcp(
    config: MCPBridgeConfig,
    repo_root: Path,
    *,
    bridge: McpBridge | None = None,
) -> None:
    """Run the workflow MCP stdio loop until EOF, then only disconnect the GUI.

    An omitted bridge creates one instance with config; an injected bridge must
    use config and may have an attached recording Transport. Assemble tools with
    build_workflow_tools, then use the shared stdio error/content projection.
    EOF never stops or kills a launched GUI. No hardware or GUI is started until
    the caller explicitly invokes the shared launch/connect lifecycle tools.
    """
    if bridge is None:
        bridge = McpBridge(config)
    tools = build_workflow_tools(config, bridge, repo_root)

    def _disconnect() -> None:
        bridge.disconnect()

    run_stdio_loop(config, tools, hooks=StdioLoopHooks(on_cleanup=_disconnect))
