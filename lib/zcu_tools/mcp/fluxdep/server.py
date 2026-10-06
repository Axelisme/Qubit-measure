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

from zcu_tools.gui.app.fluxdep.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)
from zcu_tools.mcp.core.bridge import MCPBridgeConfig  # noqa: E402
from zcu_tools.mcp.fluxdep.assembly import build_fluxdep_server  # noqa: E402

MCP_VERSION = 2

_SERVER_INSTRUCTIONS = """\
Control the Fluxdep analysis pipeline in the same GUI the user can drive.
No hardware is controlled. Use existing data only within the user's authorization.

Start with fluxdep_connect to an existing GUI, or explicitly fluxdep_launch.
Disconnect and MCP server exit leave the GUI running. There is no stop tool.

Before writes, explicitly read the required published resources on this connection:
project_info, spectrum_list, each spectrum_snapshot, selection_snapshot, fit_result.
State checks, derived pointclouds, interactive state and PNG do not establish those
observations. Writes never perform hidden reads. On stale rejection, inspect the
GUI changes and reread the relevant full resources before deciding what to do.

Pipeline:
1. Read project_info; project_setup applies identity and native paths.
2. Read collection/all current spectrum snapshots; spectrum_load loads OneTone or
   TwoTone data, or spectrum_load_processed restores a processed export. New spectra
   need explicit snapshots before editing. Names are literal, including punctuation.
3. Read spectrum_list, set the active spectrum, then read its spectrum_snapshot.
   spectrum_interactive_open(name, kind) opens line, onetone or twotone input.
4. interactive_read inspects without opening or changing focus. Open/read/command
   receipts contain identity, committed state, available commands and native image
   content. context.figure is MIME/byte metadata, not a saved file. Inactive reads
   return null context without an image. Inspect axes and units before choosing points.
5. spectrum_interactive_command uses the returned context_id, literal name and a
   plugin command with its declared params. undo restores one committed step.
   finish publishes even a zero-point completed spectrum; cancel closes input.
6. Read collection/all sources (including zero-point spectra) and selection_snapshot;
   selection_interactive_open/command edits the joint cloud. apply publishes while
   preserving editable input; cancel closes. Width is normalized radius. Follow each
   returned command schema. Receipts show numeric changes and the same captured PNG.
7. Read fit_result before fit_set_params. EJb/ECb/ELb are numeric bounds in GHz.
   Project database_path is the raw-data root; fit database_path is the search file.
   Read project, fit, collection, every source and selection before fit_search.
8. fit_search returns the app token. operation_status without token finds latest GUI
   or agent activity. operation_await(token, timeout) waits at most 30 seconds and
   never cancels on timeout. operation_cancel is a stop request, not terminal proof.
   A failed operation is outcome data, distinct from an invocation error. After
   completion reread fit_result; operation reads do not refresh any write guard.
9. export_spectrums is create-only unless overwrite is explicit. fit_export_params
   uses native JSON merge. Read their declared resources first. Failures may retain
   confirmed output prefixes; there is no rollback.

The user may take over at any time. Open reuses an eligible identity; commands
require the current identity. Terminal receipts describe the requested old identity,
not a synchronous GUI successor. To inspect the successor, explicitly interactive_read.

Tools send one RPC. They do not reconnect, retry, replay or repair stale observations.
Timeout/disconnection/image-delivery failure does not prove a mutation had no effect.
Read current state before choosing another command. Image delivery failure does not
undo accepted GUI publication. Stdio handles calls synchronously; an await occupies
this server until it returns. Transport deadlines must cover the method's budget
(35 seconds for operation_await) plus reply overhead.
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
