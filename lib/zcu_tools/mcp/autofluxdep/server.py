#!/usr/bin/env python
"""MCP server bridge for the autofluxdep-gui ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live autofluxdep GUI's ``RemoteControlAdapter``
over a single persistent TCP socket. The exposed tools are READ-ONLY: every
workflow-observation tool is generated 1:1 from the wire-method contract table
(``METHOD_SPECS``, all pure queries — the user drives the GUI); the agent-facing
lifecycle tools (``autofluxdep_launch`` / ``autofluxdep_connect`` /
``autofluxdep_disconnect``) are hand-written and fork
``script/run_autofluxdep_gui.py``.

The socket/RPC/stdio plumbing lives in the shared
:mod:`zcu_tools.mcp.core.bridge` (:class:`McpBridge` + helpers); this module
keeps only autofluxdep's config + the read-only ``send_gui_rpc`` wrapper + the
three lifecycle tools. Events are dropped (the agent uses request/reply, not
event subscription), so no ``on_event`` hook is wired.

Threading: see :mod:`zcu_tools.mcp.core.bridge`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import gettempdir
from typing import Any

# This bridge is launched standalone (``python .../server.py``), so the repo
# ``lib`` dir is not on sys.path by default. Add it so the wire-contract modules
# import cleanly.
# lib/zcu_tools/mcp/autofluxdep -> lib
_LIB_DIR = Path(__file__).resolve().parents[3]
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Fast-fail preflight: the GUI fork needs the 'gui' extra (qtpy).
for _gui_dep in ("qtpy",):
    if importlib.util.find_spec(_gui_dep) is None:
        sys.stderr.write(
            "autofluxdep-gui MCP server requires the 'gui' extra (qtpy); "
            f"'{_gui_dep}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n"
        )
        raise SystemExit(1)

# NOTE: absolute imports (NOT relative) — this module is launched as a script
# (``python .../server.py`` per .mcp.json), so it has no parent package.
from zcu_tools.gui.app.autofluxdep.services.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)
from zcu_tools.gui.app.autofluxdep.services.remote.wire_version import (  # noqa: E402
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
Observe a live autofluxdep-gui (automated flux-dependence workflow) over a TCP socket.

This bridge is READ-ONLY: the USER drives the workflow in the GUI (set up the
project + SoC, assemble the node list, set the flux sweep + flux device, Run /
Stop the sweep). The agent's job is to watch and report current state — there are
no setup / edit-node / set-flux / run / stop tools, because building the workflow
and judging the live fits need the user's eye on the GUI.

Getting started:
  1. autofluxdep_launch opens a GUI subprocess for the user (auto-connects).
     Or autofluxdep_connect to attach to a GUI the user already started.
  2. The user drives the workflow; you observe it with the read tools.
  autofluxdep_disconnect detaches the bridge without stopping the GUI. There is
  no stop tool — the agent never closes the user's GUI nor stops a running sweep.

Read tools (all pure queries):
  - autofluxdep_state_check → {has_project, has_soc, node_count, flux_count,
    has_flux_device, is_running, has_results, has_loaded_predictor,
    has_run_predictor}.
  - autofluxdep_project_info → {chip_name, qub_name, result_dir, params_path}.
  - autofluxdep_workflow_list → each placed node's {name, type, provides,
    provides_modules, requires, has_result}.
  - autofluxdep_node_cfg(name) → {name, type, knobs:{...}} for one placed node.
  - autofluxdep_result_summary → per node-with-result {name, kind, n_flux,
    n_measured, fit_summary} — how far the sweep has progressed, not the raw 2D data.

A failed call always raises; the read tools are idempotent, so retrying is safe.
"""

_CONFIG = MCPBridgeConfig(
    app_name="autofluxdep",
    tool_prefix="autofluxdep_",
    default_port=8768,
    mcp_version=MCP_VERSION,
    wire_version=MCP_WIRE_VERSION,
    server_display_name="autofluxdep-gui-control",
    server_instructions=_SERVER_INSTRUCTIONS,
    pid_file=Path(gettempdir()) / "zcu_tools_autofluxdep_gui.pid",
    log_file=Path(gettempdir()) / "zcu_tools_autofluxdep_gui_debug.log",
    run_script_name="run_autofluxdep_gui.py",
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


def tool_autofluxdep_connect(arguments: dict[str, Any]) -> str:
    port = arguments.get("port", _CONFIG.default_port)
    if not isinstance(port, int):
        raise ValueError("Invalid 'port' argument (must be integer)")
    return _BRIDGE.connect(port, arguments.get("token"))


def tool_autofluxdep_disconnect(arguments: dict[str, Any]) -> str:
    del arguments
    return _BRIDGE.disconnect()


def tool_autofluxdep_launch(arguments: dict[str, Any]) -> str:
    port = int(arguments.get("port", _CONFIG.default_port))
    token: str | None = arguments.get("token")
    auto_connect = bool(arguments.get("auto_connect", True))
    # lib/zcu_tools/mcp/autofluxdep -> repo root
    repo_root = Path(__file__).parents[4]
    return _BRIDGE.launch(repo_root, port, token, auto_connect)


# autofluxdep_stop is intentionally NOT exposed as an agent tool: the agent
# observes a GUI the user drives and must not kill the user's GUI nor stop a
# running sweep. The bridge's own stop() is used in _cleanup_on_exit so a
# server-launched GUI is not orphaned.
_NON_GENERATED_METHODS = frozenset(
    {
        # mcp<->RPC bookkeeping only; version numbers must not surface to the agent.
        "resources.versions",
    }
)

_OVERRIDE_TOOLS: dict[str, dict[str, Any]] = {
    "autofluxdep_connect": {
        "handler": tool_autofluxdep_connect,
        "description": (
            "Connect the MCP bridge to an ALREADY-RUNNING autofluxdep-gui's TCP "
            "control port (default 8768). Errors if no GUI is listening there — "
            "use autofluxdep_launch to start one. Skip this if you used "
            "autofluxdep_launch with auto_connect=true (default)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP port of a running GUI control service (default 8768)",
                },
                "token": {
                    "type": "string",
                    "description": "Optional authentication token",
                },
            },
        },
    },
    "autofluxdep_disconnect": {
        "handler": tool_autofluxdep_disconnect,
        "description": (
            "Disconnect the MCP bridge from the GUI control port. Does NOT stop "
            "the GUI process — it keeps running for the user to drive."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    "autofluxdep_launch": {
        "handler": tool_autofluxdep_launch,
        "description": (
            "Launch the autofluxdep-gui as a NEW subprocess on a TCP control port "
            "(default 8768), wait until ready, and optionally connect. Use as the "
            "first step. Errors if the port is already in use (a stale GUI). By "
            "default auto_connect=true."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "port": {
                    "type": "integer",
                    "description": "TCP control port for the GUI (default 8768)",
                },
                "token": {
                    "type": "string",
                    "description": "Optional shared auth token",
                },
                "auto_connect": {
                    "type": "boolean",
                    "default": True,
                    "description": "Call autofluxdep_connect automatically once ready (default true)",
                },
            },
        },
    },
}

_OVERRIDE_NAMES = frozenset(
    {"autofluxdep_connect", "autofluxdep_disconnect", "autofluxdep_launch"}
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
