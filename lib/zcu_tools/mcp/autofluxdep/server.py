#!/usr/bin/env python
"""MCP server bridge for the autofluxdep-gui ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live autofluxdep GUI's ``RemoteControlAdapter``
over a single persistent TCP socket. The exposed tools are READ-ONLY: every
workflow-observation tool is generated 1:1 from the wire-method contract table
(``METHOD_SPECS``, all pure queries — the user drives the GUI); the agent-facing
lifecycle tools (``autofluxdep_launch`` / ``autofluxdep_connect`` /
``autofluxdep_disconnect``) are built by the shared read-only factory and fork
``script/run_autofluxdep_gui.py``.

The whole server body (``send_gui_rpc``, the lifecycle tools, cleanup, the stdio
loop) lives in :func:`zcu_tools.mcp.core.readonly_server.build_readonly_server`;
this module keeps only autofluxdep's config + instructions + the imports the
factory needs. Events are dropped (request/reply, not event subscription).

Threading: see :mod:`zcu_tools.mcp.core.bridge`.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from tempfile import gettempdir

_BOOTSTRAP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "_standalone.py"))
_BOOTSTRAP["bootstrap_standalone_server"](
    __file__,
    required_modules=(
        (
            "qtpy",
            "autofluxdep-gui MCP server requires the 'gui' extra (qtpy); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
    ),
)

# NOTE: absolute imports (NOT relative) — this module is launched as a script
# (``python .../server.py`` per .mcp.json), so it has no parent package.
from zcu_tools.gui.app.autofluxdep.services.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)
from zcu_tools.gui.app.autofluxdep.services.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)
from zcu_tools.mcp.core.bridge import MCPBridgeConfig  # noqa: E402
from zcu_tools.mcp.core.readonly_server import (  # noqa: E402
    READONLY_MCP_VERSION,
    build_readonly_server,
)

# This MCP server's own code revision — reported (not compared) in the version
# note so an agent can confirm a reconnect picked up bridge-side edits.
MCP_VERSION = READONLY_MCP_VERSION

_SERVER_INSTRUCTIONS = """\
Observe a live autofluxdep-gui (automated flux-dependence workflow) over a TCP socket.

This bridge is READ-ONLY: the USER drives the workflow in the GUI (set up the
project + SoC, assemble the node list, set the flux sweep + flux device, Run /
Pause/Continue/Abort the sweep). The agent's job is to watch and report current state — there are
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
    has_flux_device, is_running, is_paused, next_flux_idx, run_status,
    has_results, has_loaded_predictor, has_run_predictor}.
  - autofluxdep_project_info → {chip_name, qub_name, result_dir,
    database_path, params_path}.
  - autofluxdep_workflow_list → each placed node's {name, type, enabled,
    provides, provides_modules, requires, has_result}.
  - autofluxdep_node_cfg(name) → {name, type, knobs:{...}} for one placed node.
  - autofluxdep_result_summary → per node-with-result {name, kind, n_flux,
    n_measured, fit_summary} — how far the sweep has progressed, not the raw 2D data.

A failed call always raises; the read tools are idempotent, so retrying is safe.
"""

_CONFIG = MCPBridgeConfig(
    app_name="autofluxdep",
    app_slug="autofluxdep",
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

# lib/zcu_tools/mcp/autofluxdep -> repo root
_SERVER = build_readonly_server(
    _CONFIG,
    METHOD_SPECS,
    repo_root=Path(__file__).parents[4],
    gui_name="autofluxdep-gui",
)

# Module-level aliases preserved for tests that patch the bridge / inspect tools.
_BRIDGE = _SERVER.bridge
send_gui_rpc = _SERVER.send_gui_rpc
TOOLS = _SERVER.tools
main = _SERVER.main


if __name__ == "__main__":
    main()
