#!/usr/bin/env python
"""Measure MCP stdio server, attached to a live GUI control socket."""

from __future__ import annotations

import logging
import runpy
from pathlib import Path
from tempfile import gettempdir

logger = logging.getLogger(__name__)

# Standalone entry point also works without an installed editable package.
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

from zcu_tools.gui.app.measure.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)
from zcu_tools.mcp.core.bridge import (  # noqa: E402
    McpBridge,
    MCPBridgeConfig,
    port_is_open,
    resolve_connect_port,
)
from zcu_tools.mcp.measure.assembly import build_measure_tools  # noqa: E402
from zcu_tools.mcp.measure.session import MeasureMcpSession  # noqa: E402
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
MCP_VERSION = 104

_SERVER_INSTRUCTIONS = """\
Attach to the live qubit-measure GUI with connect (no instrument is connected by
this action). An existing GUI can be attached, or launch='if_missing'/'new' can
start one. Pass token to connect if the GUI requires its control token; keep it
secret. Inspect connect.status and refresh the GUI state before acting; the
user may also be editing it. When the GUI restarts, the next connection reloads
the live catalog. Incompatible wire versions fail before an action is forwarded.

Use rpc_list(domain?) to find low-frequency methods, rpc_describe(method) for
the GUI's live parameter schema and full description, and rpc_call(method,
params) for any listed method, including those with specialized tool helpers.
Tool names are guidance, not a separate mode or permission. GUI handlers validate
arguments and return stable error reasons. Mutations are never automatically retried after disconnect,
timeout, stale_version or busy; read the current state before choosing to retry.
For tab/context/SoC guard conflicts, explicitly read tab.snapshot(tab_id),
context.snapshot, and soc.info(include_cfg=true), respectively. Summaries,
partial getters, bare versions and status do not re-snapshot those resources.
A new tab created with tab.new carries an owner-thread existence receipt.
Use tab_interact without payload to read the active plugin's committed state and
commands; send one payload={command,args} to act. This method alone has no seen
guard: GUI and agent commits use owner-loop order, and the later commit wins.
Reads preserve focus; commands follow the Analysis pane. done returns the original
analysis operation receipt, then joins its execution for result reads and image
saving. preview_active describes local preview, not committed state. Figure paths
belong to this MCP session; saved_images lists confirmed persistent image paths.
Use status for the GUI session, live operations and this session's executions.
status(execution) reads a local snapshot without reconnecting. wait(op) observes
only the GUI operation; wait(execution) also waits for result reads, image saving
and preview delivery. Both report failed outcomes as data, not query errors.
Wait timeouts stop waiting, not the operation or its background continuation.
For a registered analysis op or execution, cancel latches continuation intent even
when the GUI operation is not cancellable. It starts no further result read or
save, but retains the true outcome of any admitted save. gui_cancel describes the
GUI request separately; terminal executions return not_needed without rewriting
their outcome. Only unregistered cancel(op) retains the direct GUI hook behavior,
including operation_failed for an already failed operation.
The server disconnects and joins its workers before removing temporary PNGs on
exit. It does not close the GUI or promise to stop hardware. There are no
subscribed MCP events: read snapshots or wait on operation handles instead.
Follow the run-measure-gui skill for hardware safety and measurement workflow.
"""

_CONFIG = MCPBridgeConfig(
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
    run_script_name="run_measure_gui.py",
)


def _setup_logging() -> None:
    from zcu_tools.gui.logging_setup import setup_gui_logging

    setup_gui_logging(
        app_name="measure",
        log_root=Path(__file__).resolve().parents[4],
        group="mcp",
        extra_namespaces=("zcu_tools.mcp",),
    )


def main() -> None:
    from zcu_tools.mcp.core.stdio_server import StdioLoopHooks, run_stdio_loop

    session = MeasureMcpSession(
        _CONFIG,
        resolve_connect_port=resolve_connect_port,
        port_is_open=port_is_open,
    )
    bridge = McpBridge(_CONFIG)
    session.attach_bridge(bridge)
    context = MeasureToolContext(
        config=_CONFIG,
        session=session,
        resolve_connect_port=resolve_connect_port,
    )

    try:
        run_stdio_loop(
            _CONFIG,
            build_measure_tools(context),
            hooks=StdioLoopHooks(on_start=_setup_logging, on_error=logger.exception),
            server_version="1.1.0",
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
