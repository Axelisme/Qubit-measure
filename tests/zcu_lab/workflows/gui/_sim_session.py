"""In-process TCP substitute for the real Qt workflow composition."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from qtpy.QtWidgets import QApplication
from zcu_tools.gui.app.workflows.app import WorkflowsGuiBehavior
from zcu_tools.gui.app.workflows.remote import WorkflowRemoteControlAdapter
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.mcp.workflows.server import build_workflow_tools
from zcu_tools.resources.entry import ResultEntry

from zcu_lab.workflows.calibration import Calibrated

from ._recording_transport import WindowTransport


def wait_owner(qapp: QApplication, predicate: Callable[[], bool]) -> None:
    """Pump Qt until the public observation succeeds, or fail within 20 seconds."""
    deadline = monotonic() + 20
    while not predicate():
        assert monotonic() < deadline, "Qt workflow observation did not settle"
        qapp.processEvents()


@dataclass
class SimSession:
    """One real app and entry, with native MCP tools over an in-process transport."""

    app: WorkflowsGuiBehavior[Calibrated]
    entry: ResultEntry
    tools: ToolTable
    transport: WindowTransport

    def call(self, method: str, arguments: dict[str, object]) -> dict[str, object]:
        """Invoke one generated tool and require its ordinary JSON reply."""
        result = self.tools["workflows_workflow_" + method]["handler"](arguments)
        assert isinstance(result, dict), result
        return result

    def image(self, run_id: str, name: str) -> ToolReply:
        """Invoke the native image tool and retain its validated PNG projection."""
        result = self.tools["workflows_workflow_figure"]["handler"](
            {"run_id": run_id, "name": name}
        )
        assert isinstance(result, ToolReply), result
        return result


def connect_tools(
    adapter: WorkflowRemoteControlAdapter[Calibrated],
    qapp: QApplication,
    root: Path,
    io: ThreadPoolExecutor,
) -> tuple[ToolTable, WindowTransport, McpBridge, ClientLink]:
    """Bind real tools/RPC to the assembled app without opening a socket."""
    link = ClientLink("native-sim", token_required=False)
    adapter.on_client_open(link)

    def route(request: Request) -> dict[str, object]:
        future = io.submit(adapter.route, link, request)
        wait_owner(qapp, future.done)
        future.result(timeout=1)
        return json.loads(link.outbound.get(timeout=1))

    config = MCPBridgeConfig(
        tool_prefix="workflows_",
        server_display_name="workflows",
        server_instructions="Native simulator contract",
        app_name="workflows",
        default_port=0,
        mcp_version=1,
        wire_version=1,
        pid_file=root / "gui.pid",
        log_file=root / "gui.log",
        run_script_name="run_workflows_gui.py",
        app_slug="workflows",
    )
    transport = WindowTransport(route)
    bridge = McpBridge(config, transport=transport)
    return build_workflow_tools(config, bridge, root), transport, bridge, link
