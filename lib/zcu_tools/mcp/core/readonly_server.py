"""Factory for the observe-only dispersive and autofluxdep MCP apps.

Lifecycle tools are shared with the Fluxdep control assembly. Read forwarding
and owned-GUI cleanup retain the existing observe-only policy. Measure composes
its recipe/session machinery separately.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.lifecycle import build_lifecycle_tools
from zcu_tools.mcp.core.stdio_server import (
    StdioLoopHooks,
    ToolTable,
    assemble_tools,
    generate_tools,
    run_stdio_loop,
)

logger = logging.getLogger(__name__)

# This MCP server code revision is reported (not compared) in the version note so
# an agent can confirm a reconnect picked up bridge-side edits. Both
# read-only servers share the same revision since they share this body.
READONLY_MCP_VERSION = 1

# mcp<->RPC bookkeeping only; version numbers must not surface to the agent.
_NON_GENERATED_METHODS = frozenset({"resources.versions"})


@dataclass(frozen=True)
class ReadonlyServer:
    """The assembled pieces of one read-only MCP server.

    ``bridge`` / ``send_gui_rpc`` / ``tools`` are exposed for tests (which patch
    ``bridge.send_rpc_raw`` and inspect ``tools``); ``main`` is the stdio entry.
    """

    config: MCPBridgeConfig
    bridge: McpBridge
    send_gui_rpc: Callable[..., dict[str, Any]]
    tools: ToolTable
    main: Callable[[], None]


def build_readonly_server(
    config: MCPBridgeConfig,
    method_specs: dict[str, Any],
    repo_root: Path,
    gui_name: str,
) -> ReadonlyServer:
    """Assemble a read-only MCP server from its config + method-spec table.

    ``repo_root`` anchors the GUI launch script; pass ``Path(__file__).parents[4]``
    from the caller (``lib/zcu_tools/mcp/<app>/server.py`` -> repo root).
    ``gui_name`` is the human GUI name embedded in the lifecycle-tool prose (e.g.
    ``fluxdep-gui``). Events are dropped (READ-ONLY: no ``on_event`` hook), so the
    GUI's event stream never reaches the agent.
    """
    bridge = McpBridge(config)

    def send_gui_rpc(
        method: str, params: dict[str, Any], timeout_seconds: float = 30.0
    ) -> dict[str, Any]:
        """Issue one RPC against the GUI; raises on error or timeout."""
        resp = bridge.send_rpc_raw(method, params, timeout_seconds)
        if not resp.get("ok", False):
            err = resp.get("error", {})
            msg = f"GUI Error ({err.get('code')}): {err.get('message')}"
            reason = err.get("reason")
            if reason:
                msg += f" (reason: {reason})"
            raise RuntimeError(msg)
        return dict(resp.get("result", {}))

    overrides, override_names = build_lifecycle_tools(
        config, bridge, repo_root, gui_name
    )
    tools = assemble_tools(
        generate_tools(config, method_specs, _NON_GENERATED_METHODS, send_gui_rpc),
        overrides,
        override_names,
    )

    def _cleanup_on_exit() -> None:
        # Only stop a GUI THIS server launched; an attach-only server must not shut
        # down a GUI another process owns (the shared pid-file fallback in stop()
        # would otherwise let it kill a GUI it merely connected to).
        if not bridge.launched_gui:
            return
        # Best-effort GUI shutdown on host disconnect; swallow so a stop failure
        # never crashes the exit path, but log it so the leak is observable.
        try:
            bridge.stop(timeout_kill=True)
        except Exception:
            logger.debug("read-only bridge stop on exit failed", exc_info=True)

    def main() -> None:
        run_stdio_loop(config, tools, hooks=StdioLoopHooks(on_cleanup=_cleanup_on_exit))

    return ReadonlyServer(
        config=config,
        bridge=bridge,
        send_gui_rpc=send_gui_rpc,
        tools=tools,
        main=main,
    )


__all__ = ["READONLY_MCP_VERSION", "ReadonlyServer", "build_readonly_server"]
