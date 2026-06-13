#!/usr/bin/env python
"""MCP server bridge for the dispersive-fit-gui ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live dispersive GUI's ``RemoteControlAdapter``
over a single persistent TCP socket. The exposed tools are READ-ONLY: every
analysis method tool is generated 1:1 from the wire-method contract table
(``METHOD_SPECS``, all pure queries — the user drives the GUI); the agent-facing
lifecycle tools (``dispersive_launch`` / ``dispersive_connect`` /
``dispersive_disconnect``) are built by the shared read-only factory and fork
``script/run_dispersive_gui.py``.

The whole server body (``send_gui_rpc``, the lifecycle tools, cleanup, the stdio
loop) lives in :func:`zcu_tools.mcp.core.readonly_server.build_readonly_server`;
this module keeps only dispersive's config + instructions + the imports the
factory needs. Events are dropped (request/reply, not event subscription).

Threading: see :mod:`zcu_tools.mcp.core.bridge`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import gettempdir

# This bridge is launched standalone (``python .../server.py``), so the repo
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
# (``python .../server.py`` per .mcp.json), so it has no parent package.
from zcu_tools.gui.app.dispersive.services.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)
from zcu_tools.gui.app.dispersive.services.remote.wire_version import (  # noqa: E402
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

# lib/zcu_tools/mcp/dispersive -> repo root
_SERVER = build_readonly_server(
    _CONFIG,
    METHOD_SPECS,
    repo_root=Path(__file__).parents[4],
    gui_name="dispersive-fit-gui",
)

# Module-level aliases preserved for tests that patch the bridge / inspect tools.
_BRIDGE = _SERVER.bridge
send_gui_rpc = _SERVER.send_gui_rpc
TOOLS = _SERVER.tools
main = _SERVER.main


if __name__ == "__main__":
    main()
