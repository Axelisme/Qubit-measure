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

from zcu_tools.gui.app.main.services.remote.wire_version import (  # noqa: E402
    WIRE_VERSION as MCP_WIRE_VERSION,
)
from zcu_tools.mcp.core.bridge import (  # noqa: E402
    McpBridge,
    MCPBridgeConfig,
    _port_is_open,
    resolve_connect_port,
    run_stdio_loop,
)
from zcu_tools.mcp.measure.assembly import build_measure_tools  # noqa: E402
from zcu_tools.mcp.measure.session import MeasureMcpSession  # noqa: E402
from zcu_tools.mcp.measure.tool_context import MeasureToolContext  # noqa: E402

MCP_VERSION = 82

_SERVER_INSTRUCTIONS = """\
Attach to the live qubit-measure GUI with connect (no instrument is connected by
this action). An existing GUI can be attached, or launch='if_missing'/'new' can
start one. Pass token to connect if the GUI requires its control token; keep it
secret. Inspect connect.status and refresh the GUI state before acting; the
user may also be editing it. When the GUI restarts, the next connection reloads
the live catalog. Incompatible wire versions fail before an action is forwarded.

Use rpc_list(domain?) to find low-frequency methods, rpc_describe(method) for
the GUI's live parameter schema and full description, and rpc_call(method,
params) only for methods with exposure='rpc'. Methods bound to a specialized
tool report use_tool instead. GUI handlers validate arguments and return stable
error reasons. Mutations are never automatically retried after disconnect,
timeout, stale_version or busy; read the current state before choosing to retry.
For tab/context/SoC guard conflicts, explicitly read tab.snapshot(tab_id),
context.snapshot, and soc.info(include_cfg=true), respectively. Summaries,
partial getters, bare versions and status do not re-snapshot those resources.
A new tab created with tab.new carries an owner-thread existence receipt.
Use status for the current GUI session and all live operations, wait(op) for a
bounded outcome, and cancel(op) only when the domain operation supports it.
wait(op) reports a failed operation as data; cancel(op) on a failed operation
raises operation_failed instead of reporting success. The server drops its
socket on exit but does not close the GUI. There are no
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
    session = MeasureMcpSession(
        _CONFIG,
        resolve_connect_port=resolve_connect_port,
        port_is_open=_port_is_open,
    )
    bridge = McpBridge(_CONFIG)
    session.attach_bridge(bridge)
    context = MeasureToolContext(
        config=_CONFIG,
        session=session,
        resolve_connect_port=resolve_connect_port,
    )

    def cleanup() -> None:
        bridge.disconnect()

    run_stdio_loop(
        _CONFIG,
        build_measure_tools(context),
        on_cleanup=cleanup,
        on_start=_setup_logging,
        on_error=logger.exception,
        server_version="1.1.0",
    )


if __name__ == "__main__":
    main()
