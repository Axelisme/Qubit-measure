"""In-process TCP substitute for the real Qt workflow composition."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from qtpy.QtWidgets import QApplication
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request, parse_request
from zcu_tools.mcp.core.bridge import DeliverFn, McpBridge, MCPBridgeConfig, OnClosedFn
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.core.stdio_server import ToolTable
from zcu_tools.resources.entry import ResultEntry

from zcu_lab.workflows.gui.app import WorkflowsGuiBehavior
from zcu_lab.workflows.gui.remote import T1RemoteControlAdapter
from zcu_lab.workflows.mcp import build_workflow_tools


def wait_owner(qapp: QApplication, predicate: Callable[[], bool]) -> None:
    """Pump Qt until the public observation succeeds, or fail within 20 seconds."""
    deadline = monotonic() + 20
    while not predicate():
        assert monotonic() < deadline, "Qt workflow observation did not settle"
        qapp.processEvents()


class WindowTransport:
    """Replace TCP only; preserve bridge IDs, wire parsing and owner RPC dispatch."""

    def __init__(self, route: Callable[[Request], dict[str, object]]) -> None:
        self.route = route
        self.requests: list[Request] = []
        self.deliver_reply: DeliverFn | None = None
        self.is_open = True

    def attach(
        self, deliver_reply: DeliverFn, deliver_event: DeliverFn, on_closed: OnClosedFn
    ) -> None:
        """Bind bridge delivery; this substitute does not emit transport events."""
        self.deliver_reply = deliver_reply

    def send_line(self, payload: dict[str, object]) -> None:
        """Dispatch the original request through the real remote adapter once."""
        if not self.is_open or self.deliver_reply is None:
            raise RuntimeError("Transport is not attached")
        request = parse_request(json.loads(json.dumps(payload)))
        self.requests.append(request)
        self.deliver_reply(self.route(request))

    def close(self) -> None:
        """Disconnect MCP without cancelling the GUI run."""
        self.is_open = False


@dataclass
class SimSession:
    """One real app and entry, with native MCP tools over an in-process transport."""

    app: WorkflowsGuiBehavior
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
    adapter: T1RemoteControlAdapter,
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
