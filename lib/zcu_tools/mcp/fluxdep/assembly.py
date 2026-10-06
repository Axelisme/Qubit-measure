"""Fluxdep control tools assembled from the GUI's wire declarations."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from zcu_tools.gui.app.fluxdep.remote.method_specs import METHOD_SPECS
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.images import validated_png
from zcu_tools.mcp.core.lifecycle import build_lifecycle_tools
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import (
    StdioLoopHooks,
    ToolTable,
    assemble_tools,
    generate_tools,
    run_stdio_loop,
)


@dataclass(frozen=True)
class FluxdepServer:
    """One control server's assembled owners.

    bridge owns its GUI connection and optional launched process. tools contains
    the generated pipeline and explicit lifecycle callbacks. main runs the stdio
    loop; stdin closure disconnects without shutting down or killing the GUI.
    """

    bridge: McpBridge
    tools: ToolTable
    main: Callable[[], None]


def project_interactive_reply(reply: ToolReply) -> ToolReply:
    """Deliver a successful GUI {context, effect} receipt as text plus one PNG.

    Inactive context returns unchanged text and no images. A live/closed context
    retains identity, state and effect; figure becomes MIME/byte metadata instead
    of base64 text. Raise on base64/PNG decoding failure, without changing GUI
    state, retrying the command or storing a file. The input receipt is unchanged.
    """
    data = dict(reply.data)
    context = data["context"]
    if context is None:
        return ToolReply(data)
    context = dict(context)
    figure = context["figure"]
    image = validated_png(base64.b64decode(figure["png_b64"], validate=True))
    context["figure"] = {"mime_type": "image/png", "bytes": figure["bytes"]}
    data["context"] = context
    return ToolReply(data, images=(image,))


def build_fluxdep_server(
    config: MCPBridgeConfig,
    repo_root: Path,
    *,
    bridge: McpBridge | None = None,
) -> FluxdepServer:
    """Assemble pipeline/lifecycle tools and the stdio entry for Fluxdep.

    config supplies connection/launch settings and server instructions. repo_root
    anchors the GUI launch script. An injected bridge must use this config; tests
    may attach a recording Transport. GUI declarations own names, schemas and
    budgets. Tools never preread, retry or recreate seen/session/operation state.
    Invocation errors raise with native code/message/reason; failed operation
    outcomes remain successful data. Stdio isolation turns raised errors into
    tool-error content. A mismatched injected config raises ValueError before
    assembly. Events are not subscribed or queued.
    """
    if bridge is not None and bridge.config != config:
        raise ValueError("Injected bridge must use the supplied Fluxdep config")
    bridge = bridge if bridge is not None else McpBridge(config)

    def send_gui_rpc(
        method: str, params: dict[str, object], timeout_seconds: float = 30.0
    ) -> dict[str, object]:
        response = bridge.send_rpc_raw(method, params, timeout_seconds)
        if not response["ok"]:
            error = response["error"]
            message = f"GUI Error ({error['code']}): {error['message']}"
            if error.get("reason"):
                message += f" (reason: {error['reason']})"
            raise RuntimeError(message)
        return dict(response["result"])

    generated = generate_tools(
        config, METHOD_SPECS, frozenset({"resources.versions"}), send_gui_rpc
    )
    for method in (
        "interactive.read",
        "spectrum.interactive.open",
        "spectrum.interactive.command",
        "selection.interactive.open",
        "selection.interactive.command",
    ):
        name = config.tool_prefix + method.replace(".", "_")
        forwarder = generated[name]["handler"]
        generated[name]["handler"] = lambda arguments, forwarder=forwarder: (
            project_interactive_reply(ToolReply(forwarder(arguments)))
        )
    lifecycle, names = build_lifecycle_tools(config, bridge, repo_root, "fluxdep-gui")
    tools = assemble_tools(generated, lifecycle, names)

    def cleanup() -> None:
        bridge.disconnect()

    def main() -> None:
        run_stdio_loop(config, tools, hooks=StdioLoopHooks(on_cleanup=cleanup))

    return FluxdepServer(bridge, tools, main)
