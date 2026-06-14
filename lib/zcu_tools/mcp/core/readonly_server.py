"""Factory for the read-only GUI-bridge MCP servers.

The three observe-only apps (``fluxdep`` / ``dispersive`` / ``autofluxdep``) ship
near-identical ``server.py`` modules: the same ``send_gui_rpc`` error wrapper, the
same three lifecycle tools (``<app>_connect`` / ``<app>_disconnect`` /
``<app>_launch``), the same ``resources.versions`` skip, the same cleanup + main.
They differ only in their :class:`MCPBridgeConfig` (name / prefix / port / versions
/ instructions / file paths) and which ``METHOD_SPECS`` table they generate read
tools from.

This factory owns that shared body. A read-only ``server.py`` becomes: inject the
lib path + qtpy preflight (so its method-spec imports resolve), build its config,
then ``SERVER = build_readonly_server(config, METHOD_SPECS)`` and
``main = SERVER.main``.

``measure-gui`` does NOT use this factory: it adds an optimistic-concurrency
guard, operation tracking, a diagnostic piggyback channel and hand-written tools,
so it composes the bridge directly. This factory is deliberately scoped to the
observe-only apps — no mutating tools, no stop tool (the agent never closes the
user's GUI; the bridge's own ``stop`` runs only in cleanup so a server-launched
GUI is not orphaned).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zcu_tools.mcp.core.bridge import (
    McpBridge,
    MCPBridgeConfig,
    Tool,
    ToolTable,
    assemble_tools,
    generate_tools,
    resolve_connect_port,
    run_stdio_loop,
)

logger = logging.getLogger(__name__)

# This MCP server code revision is reported (not compared) in the version note so
# an agent can confirm a reconnect picked up bridge-side edits. All three
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


def _make_lifecycle_tools(
    config: MCPBridgeConfig, bridge: McpBridge, repo_root: Path, gui_name: str
) -> tuple[ToolTable, frozenset[str]]:
    """Build the three hand-written lifecycle tools (connect/disconnect/launch).

    ``gui_name`` is the human GUI name used in the tool prose (e.g. ``fluxdep-gui``,
    ``dispersive-fit-gui``) — distinct from ``config.server_display_name`` (the MCP
    ``serverInfo`` name).

    ``<app>_stop`` is intentionally absent: the agent observes a user-driven GUI
    and must not kill it. The bridge's own ``stop`` is used in cleanup instead.
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
        return bridge.launch(repo_root, p, token, auto_connect)

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

    overrides, override_names = _make_lifecycle_tools(
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
        run_stdio_loop(config, tools, on_cleanup=_cleanup_on_exit)

    return ReadonlyServer(
        config=config,
        bridge=bridge,
        send_gui_rpc=send_gui_rpc,
        tools=tools,
        main=main,
    )


__all__ = ["READONLY_MCP_VERSION", "ReadonlyServer", "build_readonly_server"]
