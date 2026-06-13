#!/usr/bin/env python
"""MCP server bridge for the fluxdep-gui ``RemoteControlAdapter``.

Communicates with an MCP host (Gemini / Claude / VS Code) via stdio JSON-RPC
2.0, and forwards calls to the live fluxdep GUI's ``RemoteControlAdapter`` over a
single persistent TCP socket. The exposed tools are READ-ONLY: every analysis
method tool is generated 1:1 from the wire-method contract table (``METHOD_SPECS``,
all pure queries — the user drives the GUI); the agent-facing lifecycle tools
(``fluxdep_launch`` / ``fluxdep_connect`` / ``fluxdep_disconnect``) are built by
the shared read-only factory and fork ``script/run_fluxdep_gui.py``.

The whole server body (``send_gui_rpc``, the lifecycle tools, cleanup, the stdio
loop) lives in :func:`zcu_tools.mcp.core.readonly_server.build_readonly_server`;
this module keeps only fluxdep's config + instructions + the imports the factory
needs. Events are dropped (the agent uses request/reply, not event subscription).

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
# lib/zcu_tools/mcp/fluxdep -> lib
_LIB_DIR = Path(__file__).resolve().parents[3]
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Fast-fail preflight: the GUI fork needs the 'gui' extra (qtpy).
for _gui_dep in ("qtpy",):
    if importlib.util.find_spec(_gui_dep) is None:
        sys.stderr.write(
            "fluxdep-gui MCP server requires the 'gui' extra (qtpy); "
            f"'{_gui_dep}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n"
        )
        raise SystemExit(1)

# NOTE: absolute imports (NOT relative) — this module is launched as a script
# (``python .../server.py`` per .mcp.json), so it has no parent package.
from zcu_tools.gui.app.fluxdep.services.remote.method_specs import (  # noqa: E402
    METHOD_SPECS,
)
from zcu_tools.gui.app.fluxdep.services.remote.wire_version import (  # noqa: E402
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
Observe a live fluxdep-gui (fluxonium flux-dependence analysis) over a TCP socket.

This bridge is READ-ONLY: the USER drives the analysis in the GUI (load spectra,
pick half/integer flux lines, select spectral points, cross-spectrum filter, run
the database fit, export). The agent's job is to watch and report current state —
there are no load / align / point-pick / select / fit / export tools, because
point-picking and axis-orientation judgement need the user's eye on the preview.

Getting started:
  1. fluxdep_launch opens a GUI subprocess for the user (auto-connects the bridge).
     Or fluxdep_connect to attach to a GUI the user already started.
  2. The user does the analysis in the GUI; you observe it with the read tools.
  fluxdep_disconnect detaches the bridge without stopping the GUI. There is no
  stop tool — the agent never closes the user's GUI.

Read tools (all pure queries):
  - fluxdep_state_check → {has_project, spectrum_count, has_active}.
  - fluxdep_project_info → {chip_name, qub_name, result_dir, database_path}.
  - fluxdep_spectrum_list → each loaded spectrum's {name, spec_type, aligned,
    points_selected} (i.e. how far the user has taken each spectrum).
  - fluxdep_selection_pointcloud → the joint {fluxs, freqs} cloud assembled from
    every spectrum's selected points (freqs in GHz).
  - fluxdep_fit_result → {has_result, params:{EJ,EC,EL} or null, database_path,
    EJb, ECb, ELb, transitions, r_f, sample_f} — the user's fit inputs + result.

A failed call always raises; the read tools are idempotent, so retrying a read is
safe.
"""

_CONFIG = MCPBridgeConfig(
    app_name="fluxdep",
    app_slug="fluxdep",
    tool_prefix="fluxdep_",
    default_port=8766,
    mcp_version=MCP_VERSION,
    wire_version=MCP_WIRE_VERSION,
    server_display_name="fluxdep-gui-control",
    server_instructions=_SERVER_INSTRUCTIONS,
    pid_file=Path(gettempdir()) / "zcu_tools_fluxdep_gui.pid",
    log_file=Path(gettempdir()) / "zcu_tools_fluxdep_gui_debug.log",
    run_script_name="run_fluxdep_gui.py",
)

# lib/zcu_tools/mcp/fluxdep -> repo root
_SERVER = build_readonly_server(
    _CONFIG, METHOD_SPECS, repo_root=Path(__file__).parents[4], gui_name="fluxdep-gui"
)

# Module-level aliases preserved for tests that patch the bridge / inspect tools.
_BRIDGE = _SERVER.bridge
send_gui_rpc = _SERVER.send_gui_rpc
TOOLS = _SERVER.tools
main = _SERVER.main


if __name__ == "__main__":
    main()
