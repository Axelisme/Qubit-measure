"""Measure MCP tools-lifecycle override tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zcu_tools.mcp.measure.tool_context import (
    _BRIDGE,
    _CONFIG,
    _SESSION,
    MeasureToolContext,
    _assemble_overview,
    bind_context,
    resolve_connect_port,
)


def tool_gui_connect(arguments: dict[str, Any]) -> dict[str, Any]:
    # connect attaches to a GUI that is ALREADY running (launch starts a new one),
    # so a missing GUI is the error case here. Omitting 'port' auto-discovers the
    # running GUI via its session file (covers the ephemeral-fallback case where
    # it is not on 8765), then falls back to the agreed-upon 8765.
    requested = arguments.get("port")
    if requested is not None and not isinstance(requested, int):
        raise ValueError("Invalid 'port' argument (must be integer)")
    port = resolve_connect_port(_CONFIG, requested)
    note = _BRIDGE.connect(port, arguments.get("token"))
    _SESSION.initialize_event_stream()
    # Fold the situational overview into the connect reply so attaching alone
    # gives the agent the current picture (the same data gui_overview returns),
    # saving a follow-up probe. The socket is live by here, so the fan-out reads
    # resolve against the just-attached GUI.
    return {"note": note, "overview": _assemble_overview()}


def tool_gui_disconnect(arguments: dict[str, Any]) -> dict[str, Any]:
    del arguments
    note = _BRIDGE.disconnect()
    # App-specific housekeeping: drop any buffered diagnostics — they belong to
    # the connection that just closed.
    _SESSION.clear_pending()
    return {"note": note}


def tool_gui_launch(arguments: dict[str, Any]) -> dict[str, Any]:
    port = int(arguments.get("port", _CONFIG.default_port))
    token: str | None = arguments.get("token")
    auto_connect = bool(arguments.get("auto_connect", True))
    clean = bool(arguments.get("clean", False))
    # lib/zcu_tools/mcp/measure -> repo root
    repo_root = Path(__file__).parents[4]
    # clean → run_measure_gui --clean (skip restoring the persisted session).
    extra_args = ["--clean"] if clean else None
    note = _BRIDGE.launch(repo_root, port, token, auto_connect, extra_args=extra_args)
    # Fold the situational overview only when auto_connect actually attached the
    # bridge — the fan-out reads need a live socket. With auto_connect=false the
    # GUI is up but not yet attached, so there is no live state to read.
    if _BRIDGE.is_connected:
        _SESSION.initialize_event_stream()
        return {"note": note, "overview": _assemble_overview()}
    return {"note": note}


def tool_gui_stop(arguments: dict[str, Any]) -> dict[str, Any]:
    # Graceful close over the existing RPC channel (app.shutdown runs the GUI's
    # normal window-close path on its main thread, no OS signal), then await /
    # optionally force-kill. timeout_kill defaults False here (measure-gui prefers
    # leaving a slow-closing GUI alone for a retry rather than killing it).
    timeout = float(arguments.get("timeout", 10.0))
    timeout_kill = bool(arguments.get("timeout_kill", False))
    result = _BRIDGE.stop(
        timeout=timeout, timeout_kill=timeout_kill, shutdown_rpc="app.shutdown"
    )
    # The bridge's disconnect does not clear measure-gui's diagnostic queue; do it
    # here so a later session does not see the previous one's buffered messages.
    _SESSION.clear_pending()
    # Branch on the bridge's machine-readable outcome (no prose string-matching).
    return {"stopped": result["exited"], "note": result["note"]}


OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "gui_bridge_connect": {
        "handler": tool_gui_connect,
        "description": (
            "Attach the MCP control BRIDGE to an already-running GUI — NOT the "
            "SoC (gui_soc_connect) nor an instrument (gui_device_connect). "
            "Attaches to an ALREADY-RUNNING GUI's TCP control port. "
            "OPTIONAL: the first gui_* call already auto-attaches to the running "
            "GUI via session discovery — call this only to pin a specific 'port' or "
            "pass a 'token'. Omit 'port' to auto-discover the running GUI via its "
            "session file (covers the case where it fell back off port 8765), "
            "falling back to 8765 if none is found. Errors if no GUI is listening — "
            "use gui_launch to start one. Returns {note, overview}: 'overview' is "
            "the same situational picture gui_overview returns, so attaching gives "
            "you the current state in one call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": (
                        "TCP port of a running GUI control service. Omit to "
                        "auto-discover (then fall back to 8765)."
                    ),
                },
                "token": {
                    "type": "string",
                    "description": "Optional authentication token",
                },
            },
        },
    },
    "gui_bridge_detach": {
        "handler": tool_gui_disconnect,
        "description": (
            "Detach the MCP control bridge (closes socket only; does NOT stop "
            "the GUI — use gui_stop)."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "gui_launch": {
        "handler": tool_gui_launch,
        "description": (
            "Launch the qubit-measure GUI as a NEW subprocess on TCP control "
            "port 'port' (default 8765), wait until it is ready, and optionally "
            "connect. Use this as the first step to start a session. Errors if "
            "the port is already in use (a stale GUI still running) — stop it "
            "first (gui_stop) or pass a different port; this avoids silently "
            "attaching to old code. By default auto_connect=true so the bridge "
            "is attached automatically (gui_bridge_connect). Returns {note} — "
            "plus 'overview' (the same picture gui_overview returns) when "
            "auto_connect attached the bridge."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP control port for the GUI (default 8765)",
                },
                "token": {
                    "type": "string",
                    "description": "Optional shared auth token (also passed to gui_bridge_connect if auto_connect=true)",
                },
                "auto_connect": {
                    "type": "boolean",
                    "default": True,
                    "description": "Attach the bridge automatically (gui_bridge_connect) once port is ready (default true)",
                },
                "clean": {
                    "type": "boolean",
                    "default": False,
                    "description": "Start without restoring the previous persisted session (gui_state_v1.json is left untouched at startup; a normal close still flushes over it). Default false.",
                },
            },
        },
    },
    "gui_stop": {
        "handler": tool_gui_stop,
        "description": (
            "Stops ONLY a GUI this server launched; a connect-only session has "
            "nothing to stop. Stops the GUI started by gui_launch, then "
            "disconnects the MCP socket. Closes gracefully via the app.shutdown "
            "RPC (the GUI's normal window-close: persist session, disconnect "
            "devices, cleanup) — no OS kill, cross-platform. Waits up to "
            "'timeout' s for it to exit. Returns {stopped, note}: 'stopped' is "
            "true once the process is gone (graceful exit or force-kill), false "
            "when a graceful close timed out and was left running (re-run to "
            "retry); timeout_kill=true force-kills on timeout."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait for graceful exit (default 10)",
                },
                "timeout_kill": {
                    "type": "boolean",
                    "description": (
                        "Force-kill the process if it has not exited within "
                        "'timeout' (default false — leave it running and report)"
                    ),
                },
            },
        },
    },
}


def build_override_tools(ctx: MeasureToolContext) -> dict[str, dict[str, Any]]:
    bind_context(ctx)
    return OVERRIDE_TOOLS
