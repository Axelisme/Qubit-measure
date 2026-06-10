#!/usr/bin/env python
"""MCP server bridge for the dispersive-fit-gui ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live dispersive GUI's ``RemoteControlAdapter``
over a single persistent TCP socket. The exposed tools are READ-ONLY: every
analysis method tool is generated 1:1 from the wire-method contract table
(``METHOD_SPECS``, all pure queries — the user drives the GUI); the agent-facing
lifecycle tools (``dispersive_launch`` / ``dispersive_connect`` /
``dispersive_disconnect``) are hand-written and fork
``script/run_dispersive_gui.py``.

The socket/RPC/stdio plumbing lives in the shared
:mod:`zcu_tools.mcp.core.bridge` (:class:`McpBridge` + helpers); this module
keeps only dispersive's config + the read-only ``send_gui_rpc`` wrapper + the
three lifecycle tools. Events are dropped (the agent uses request/reply, not event
subscription), so no ``on_event`` hook is wired.

Threading: see :mod:`zcu_tools.mcp.core.bridge`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Optional

# This bridge is launched standalone (``python .../mcp_server.py``), so the repo
# ``lib`` dir is not on sys.path by default. Add it so the wire-contract modules
# import cleanly.
# lib/zcu_tools/mcp/dispersive -> lib
_LIB_DIR = Path(__file__).resolve().parents[3]
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Fast-fail preflight: the GUI fork needs the 'gui' extra (qtpy).
for _gui_dep in ("qtpy",):
    if importlib.util.find_spec(_gui_dep) is None:
        sys.stderr.write(
            "dispersive-fit-gui MCP server requires the 'gui' extra (qtpy); "
            f"'{_gui_dep}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n"
        )
        raise SystemExit(1)

# NOTE: absolute imports (NOT relative) — this module is launched as a script
# (``python .../mcp_server.py`` per .mcp.json), so it has no parent package.
from zcu_tools.gui.app.dispersive.services.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)
from zcu_tools.gui.app.dispersive.services.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)
from zcu_tools.mcp.core.bridge import (  # noqa: E402
    McpBridge,
    MCPBridgeConfig,
    assemble_tools,
    generate_tools,
    run_stdio_loop,
)

# This MCP server's own code revision — reported (not compared) in the version
# note so an agent can confirm a reconnect picked up bridge-side edits.
MCP_VERSION = 1

_SERVER_INSTRUCTIONS = """\
Observe a live dispersive-fit-gui (fluxonium dispersive-shift analysis) over a TCP socket.

This bridge is READ-ONLY: the USER drives the analysis in the GUI (load the
fluxonium fit from params.json, load a one-tone spectrum, preprocess it, manually
tune the coupling g and resonator frequency bare_rf until the predicted lines match
the spectrum, and export the dispersive section). The agent's job is to watch and
report current state — there are no load / preprocess / tune / export tools, because
the tuning and the judgement of when it matches need the user's eye on the plot.

The fit inputs come from fluxdep-gui: dispersive reads the ``fluxdep_fit`` section of
params.json (EJ/EC/EL + flux alignment) and writes a ``dispersive`` section
(g, bare_rf). Run fluxdep-gui first if the inputs are missing.

Getting started:
  1. dispersive_launch opens a GUI subprocess for the user (auto-connects the bridge).
     Or dispersive_connect to attach to a GUI the user already started.
  2. The user does the analysis in the GUI; you observe it with the read tools.
  dispersive_disconnect detaches the bridge without stopping the GUI. There is no
  stop tool — the agent never closes the user's GUI.

Read tools (all pure queries):
  - dispersive_state_check → {has_project, has_fit_inputs, has_onetone,
    has_preprocess, has_result} — how far the user has taken the pipeline.
  - dispersive_project_info → {chip_name, qub_name, result_dir, database_path}.
  - dispersive_fit_inputs_info → {has_inputs, params:{EJ,EC,EL} or null, flux_half,
    flux_int, flux_period, bare_rf_seed} — the fluxdep_fit handoff.
  - dispersive_preprocess_status → {has_preprocess, n_flux, n_freq, edelay}.
  - dispersive_fit_result → {has_result, g, bare_rf, res_dim} — the user's
    accepted tuning result.

A failed call always raises; the read tools are idempotent, so retrying a read is
safe.
"""

_CONFIG = MCPBridgeConfig(
    app_name="dispersive",
    tool_prefix="dispersive_",
    default_port=8767,
    mcp_version=MCP_VERSION,
    wire_version=MCP_WIRE_VERSION,
    server_display_name="dispersive-fit-gui-control",
    server_instructions=_SERVER_INSTRUCTIONS,
    pid_file=Path(gettempdir()) / "zcu_tools_dispersive_gui.pid",
    log_file=Path(gettempdir()) / "zcu_tools_dispersive_gui_debug.log",
    run_script_name="run_dispersive_gui.py",
)

# One bridge per process. Events are dropped (READ-ONLY: no on_event hook).
_BRIDGE = McpBridge(_CONFIG)


def send_gui_rpc(
    method: str, params: dict[str, Any], timeout_seconds: float = 30.0
) -> dict[str, Any]:
    """Issue one RPC against the GUI; raises on error or timeout."""
    resp = _BRIDGE.send_rpc_raw(method, params, timeout_seconds)
    if not resp.get("ok", False):
        err = resp.get("error", {})
        msg = f"GUI Error ({err.get('code')}): {err.get('message')}"
        reason = err.get("reason")
        if reason:
            msg += f" (reason: {reason})"
        raise RuntimeError(msg)
    return dict(resp.get("result", {}))


# ---------------------------------------------------------------------------
# Hand-written lifecycle tools (thin wrappers over the bridge)
# ---------------------------------------------------------------------------


def tool_dispersive_connect(arguments: dict[str, Any]) -> str:
    port = arguments.get("port", _CONFIG.default_port)
    if not isinstance(port, int):
        raise ValueError("Invalid 'port' argument (must be integer)")
    return _BRIDGE.connect(port, arguments.get("token"))


def tool_dispersive_disconnect(arguments: dict[str, Any]) -> str:
    del arguments
    return _BRIDGE.disconnect()


def tool_dispersive_launch(arguments: dict[str, Any]) -> str:
    port = int(arguments.get("port", _CONFIG.default_port))
    token: str | None = arguments.get("token")
    auto_connect = bool(arguments.get("auto_connect", True))
    # lib/zcu_tools/mcp/dispersive -> repo root
    repo_root = Path(__file__).parents[4]
    return _BRIDGE.launch(repo_root, port, token, auto_connect)


# dispersive_stop is intentionally NOT exposed as an agent tool: the agent
# observes a GUI the user drives and must not kill the user's GUI. The bridge's
# own stop() is used in _cleanup_on_exit so a server-launched GUI is not orphaned.
_NON_GENERATED_METHODS = frozenset(
    {
        # mcp<->RPC bookkeeping only; version numbers must not surface to the agent.
        "resources.versions",
    }
)

_OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "dispersive_connect": {
        "handler": tool_dispersive_connect,
        "description": (
            "Connect the MCP bridge to an ALREADY-RUNNING dispersive-fit-gui's "
            "TCP control port (default 8767). Errors if no GUI is listening "
            "there — use dispersive_launch to start one. Skip this if you used "
            "dispersive_launch with auto_connect=true (default)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP port of a running GUI control service (default 8767)",
                },
                "token": {
                    "type": "string",
                    "description": "Optional authentication token",
                },
            },
        },
    },
    "dispersive_disconnect": {
        "handler": tool_dispersive_disconnect,
        "description": (
            "Disconnect the MCP bridge from the GUI control port. Does NOT stop "
            "the GUI process — it keeps running for the user to drive."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "dispersive_launch": {
        "handler": tool_dispersive_launch,
        "description": (
            "Launch the dispersive-fit-gui as a NEW subprocess on a TCP control "
            "port (default 8767), wait until ready, and optionally connect. Use "
            "as the first step. Errors if the port is already in use (a stale "
            "GUI). By default auto_connect=true."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP control port for the GUI (default 8767)",
                },
                "token": {
                    "type": "string",
                    "description": "Optional shared auth token",
                },
                "auto_connect": {
                    "type": "boolean",
                    "default": True,
                    "description": "Call dispersive_connect automatically once ready (default true)",
                },
            },
        },
    },
}

_OVERRIDE_NAMES = frozenset(
    {"dispersive_connect", "dispersive_disconnect", "dispersive_launch"}
)


TOOLS: dict[str, dict[str, Any]] = assemble_tools(
    generate_tools(_CONFIG, METHOD_SPECS, _NON_GENERATED_METHODS, send_gui_rpc),
    _OVERRIDE_TOOLS,
    _OVERRIDE_NAMES,
)


def _cleanup_on_exit() -> None:
    try:
        _BRIDGE.stop(timeout_kill=True)
    except Exception:
        pass


def main() -> None:
    run_stdio_loop(_CONFIG, TOOLS, on_cleanup=_cleanup_on_exit)


if __name__ == "__main__":
    main()
