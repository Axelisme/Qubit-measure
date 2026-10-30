#!/usr/bin/env python
"""Measure MCP stdio server, attached to a live GUI control socket."""

from __future__ import annotations

import logging
import runpy
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.lab import MeasureMcpLab

logger = logging.getLogger(__name__)

# Validate GUI dependencies before assembling the injected server.
_BOOTSTRAP = runpy.run_path(str(Path(__file__).resolve().parents[1] / "_standalone.py"))
_BOOTSTRAP["bootstrap_standalone_server"](
    __file__,
    required_modules=(
        (
            "qtpy",
            "measure-gui MCP server requires the 'gui' extra (qtpy + PyQt6); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
        (
            "PyQt6",
            "measure-gui MCP server requires the 'gui' extra (qtpy + PyQt6); "
            "'{module}' is missing. Rebuild the environment with:\n"
            "    uv sync --extra gui\n",
        ),
    ),
)

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede GUI imports.
from zcu_tools.gui.app.measure.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede bridge imports.
from zcu_tools.mcp.core.bridge import (  # noqa: E402
    McpBridge,
    MCPBridgeConfig,
    port_is_open,
    resolve_connect_port,
)

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede tool assembly.
from zcu_tools.mcp.measure.assembly import build_measure_tools  # noqa: E402

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede session imports.
from zcu_tools.mcp.measure.session import MeasureMcpSession  # noqa: E402

# tolerated: Standalone lib-path setup and GUI dependency preflight must precede tool context imports.
from zcu_tools.mcp.measure.tool_context import MeasureToolContext  # noqa: E402

# v89: forward GUI-owned guard requests once and expose observed operation state.
# v90: fixed cfg/library tools use shared drafts and report partial library edits.
# v91: run and analyze tools share GUI operations, bounded waits and complete results.
# v92: writeback forwards preview and explicit-item batch writes to the GUI owner.
# v93: artifact save, guarded tab close and graceful shutdown tools.
# v94: fixed tab_interact tool over the shared GUI plugin session.
# v95: finalized shared-state workflow guidance and interactive concurrency instructions.
# v96: the project tool applies through the project.apply GUI method.
# v98: tab_run requires and forwards the caller's explicit cfg ref once.
# v99: generic RPC accepts all public methods, including tool-backed commands.
# v100: accept writes all current Primary and Post candidates with partial progress.
# v103: Lookback recipe execution and shared cancel/finish-early control.
# v104: Recipe interaction delivery and post-Run result provenance checks.
# v105: Onetone recipes with GUI-owned ranges and source-bound Run preview.
# v106: Two-tone spectrum and Rabi recipes with explicit frequency source precedence.
# v107: recipe-first fixed tools, shared analysis/control and complete public RPC.
# v108: analysis results retain invalid paths from the GUI's wire projection.
# v109: flux unit assertions and explicit native coordinate opt-in for FakeDevice.
# v111: injected generator recipes, captured questions and explicit answer/writeback.
# v112: status projects the opened EntryInfo rather than chip/qubit identity.
# v113: bounded partial device_setup over complete Info device.setup RPC.
# v114: integration revision above the base's device_setup release.
# v115: read-only predictor inputs and readiness without an installed predictor.
# v116: explicit working-point selection and saved writeback destinations.
# v117: holder/ref-free cfg integrated above the working-point and UI releases.
# v122: predictor RPC uses entry-local filenames or independent inline params.
# v123: actionable launch collisions and explicit raw-save analysis observations.
# v124: Async declarations, request-scoped answers, full owner evidence and scoped cancel.
# v126: Queryable writeback receipts, complete cfg preflight and source-bound analysis recovery.
MCP_VERSION = 126

_SERVER_INSTRUCTIONS = """\
Attach with connect and inspect connect.status before acting. Launch a GUI only
with authorization; startup scripts may connect hardware. Keep tokens secret.

Prefer the recipe matching the measurement goal; read its schema and
recipe_guide(adapter). Analyze existing data with tab_analyze and tab_interact.
For other setup, cfg, save, writeback or diagnosis, use rpc_list, rpc_describe,
then rpc_call. All public catalog methods remain callable.

Recipes admit one nonterminal execution. Calls and answer wait up to 300 seconds;
timeout does not cancel. Keep execution IDs. status(execution) reads progress;
detail="full" adds captured native evidence. wait(execution) includes downstream
reads, saves and image delivery; wait(op) observes only the GUI operation.
Terminal replies retain execution, status, report and runs; failed replies add
failure.phase and failure.reason. needs_parameters returns missing paths and
outcome=null; parameter_errors supplies owner paths/reasons. pending_question
carries request_id, evidence and answer_schema: use answer with that request_id.
Answering does not write; only explicit recipe accept_writeback writes.
Interactive pending={tab,stage,op} continues through tab_interact on that tab.
cancel requests stop; use wait/status to confirm completion.

Before GUI work, launch, timeout recovery or user takeover, read
.agents/skills/run-measure-gui/SKILL.md. It covers authorization, control,
execution tracking and safe continuation. Never automatically replay a mutation.
For measurement goals, policy, rounds and judgement, read
.agents/skills/measure-task/SKILL.md.

Before point-file edits, Library template creation, tab cfg editing or detailed
receipt interpretation, read lib/zcu_tools/mcp/measure/README.md.
Server exit disconnects; it does not close the GUI or promise to stop hardware.
"""


def _setup_logging() -> None:
    from zcu_tools.gui.logging_setup import setup_gui_logging

    setup_gui_logging(
        app_name="measure",
        log_root=Path(__file__).resolve().parents[4],
        group="mcp",
        extra_namespaces=("zcu_tools.mcp",),
    )


def main(lab: MeasureMcpLab, *, gui_module: str) -> None:
    """Register lab components and run stdio tools from the lab's async recipes.

    lab supplies a component registrar and recipe declarations; registration and
    invalid definitions fail before stdio dispatch or GUI operations. gui_module
    is the caller's Python module launched from the repository root. On EOF/error,
    disconnect and drain session workers/PNGs; do not close the GUI or hardware.
    """
    from zcu_tools.mcp.core.stdio_server import StdioLoopHooks, run_stdio_loop
    from zcu_tools.resources.entry.registry import component_registry

    lab.register_components(component_registry)
    recipes = lab.recipes
    config = MCPBridgeConfig(
        app_name="gui",
        app_slug="measure",
        tool_prefix="",
        default_port=8765,
        mcp_version=MCP_VERSION,
        wire_version=MCP_WIRE_VERSION,
        server_display_name="qubit-measure-control",
        server_instructions=_SERVER_INSTRUCTIONS,
        pid_file=Path(gettempdir()) / "zcu_tools_gui.pid",
        log_file=Path(gettempdir()) / "zcu_tools_gui_debug.log",
        gui_module=gui_module,
    )
    session = MeasureMcpSession(
        config,
        recipes=recipes,
        resolve_connect_port=resolve_connect_port,
        port_is_open=port_is_open,
    )
    bridge = McpBridge(config)
    session.attach_bridge(bridge)
    context = MeasureToolContext(
        config=config,
        session=session,
        resolve_connect_port=resolve_connect_port,
    )

    try:
        run_stdio_loop(
            config,
            build_measure_tools(context, recipes=recipes),
            hooks=StdioLoopHooks(
                on_start=_setup_logging,
                on_error=logger.exception,
            ),
            server_version="1.1.1",
        )
    finally:
        session.close()
