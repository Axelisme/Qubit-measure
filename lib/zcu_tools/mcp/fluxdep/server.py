#!/usr/bin/env python
"""Stdio MCP entry for the Fluxdep GUI's analysis-control pipeline.

The GUI's declarations generate schemas and time budgets. The control assembly
shares lifecycle, forwarding, PNG delivery and stdio mechanisms without copying
GUI guards, interactive owners or search operations. Events are not subscribed.
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
            "fluxdep-gui MCP server requires the 'gui' extra (qtpy); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
    ),
)

# tolerated: Wire version imports must follow lib setup and qtpy preflight.
from zcu_tools.gui.app.fluxdep.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)

# tolerated: Bridge imports must follow standalone lib setup and qtpy preflight.
from zcu_tools.mcp.core.bridge import MCPBridgeConfig  # noqa: E402

# tolerated: Control assembly needs standalone lib setup and qtpy preflight first.
from zcu_tools.mcp.fluxdep.assembly import build_fluxdep_server  # noqa: E402

MCP_VERSION = 6  # native command domains and Filter parameter differences

_SERVER_INSTRUCTIONS = """\
Drive Fluxdep flux-dependence analysis in the GUI the user also sees. No hardware
is controlled; raw data is read-only. Stay within the user's data authorization.

fluxdep_connect attaches to a running GUI; fluxdep_launch starts one only when
explicitly authorized. Their replies give project paths, spectrum stages and
pending drafts, fit freshness and control identity. Disconnect leaves the GUI running.

Tools express decision points; a stage command enters that spectrum's stage:
- fluxdep_spectrum_load / fluxdep_spectrum_snapshot: inspect spectrum evidence.
  fluxdep_spectrum_list summarizes all spectra, Filter counts and fit freshness;
  fluxdep_spectrum_remove removes data.
- fluxdep_align: align symmetry axes (J2); fluxdep_sample: sample visible lines (J3).
- fluxdep_fit: set search inputs, search to completion, then inspect params,
  per-transition residuals and the shared GUI Show overlay (J4-J5).
  fluxdep_database_info describes a search database; fluxdep_fit_result revisits the
  recorded result. Stale, absent or quality-less results return no overlay.
- fluxdep_filter: edit the joint cloud (J6), then rerun fit.
- fluxdep_export_spectrums / fluxdep_spectrum_load_processed: checkpoint and restore
  sampled spectra, without Filter masks. fluxdep_fit_export_params writes the selected entry/component predictor, setup and ledger.

Edits remain drafts until finish. Finish publishes and returns that stage's evidence
image; other edits return counts, ranges and changes. Field changes are [before, after];
point/mask changes are [added, removed]. invalidated names the publication's downstream
results; discarded names a same-spectrum draft dropped on entering another stage.
image=true requests draft evidence for edits. Commands, parameters, ranges and
transition definitions are in the tool descriptions, not in each reply.

Only the control holder writes. fluxdep_control_status / fluxdep_control_acquire /
fluxdep_control_release inspect or transfer control; the user can reclaim in the GUI.
After a rejection or an unconfirmed reply, read fluxdep_spectrum_list and, when needed,
fluxdep_spectrum_snapshot before choosing the next action. Timeout or image failure may
follow publication; do not automatically replay. auto_align and fluxdep_fit wait until
terminal without an overall limit; this stdio server handles one call at a time.
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

_SERVER = build_fluxdep_server(_CONFIG, repo_root=Path(__file__).parents[4])
TOOLS = _SERVER.tools
main = _SERVER.main

if __name__ == "__main__":
    main()
