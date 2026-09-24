"""Recording wire adapter for public measure MCP contracts."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig, ToolTable
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import MeasureMcpSession
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

RpcResponder = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass
class WireTransport:
    responder: RpcResponder | None = None
    replies: dict[str, dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]] = (
        field(default_factory=dict)
    )
    sent: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    deliver_reply: Callable[[dict[str, Any]], None] | None = None
    deliver_event: Callable[[dict[str, Any]], None] | None = None
    is_open: bool = True

    def attach(
        self,
        deliver_reply: Callable[[dict[str, Any]], None],
        deliver_event: Callable[[dict[str, Any]], None],
        on_closed: Callable[[], None],
    ) -> None:
        self.deliver_reply = deliver_reply
        self.deliver_event = deliver_event

    def send_line(self, payload: dict[str, Any]) -> None:
        method, params = payload["method"], payload["params"]
        self.sent.append((method, params))
        if method in self.replies:
            response = self.replies[method]
            reply = response(params) if callable(response) else response
        elif method == "resources.versions":
            reply = {"ok": True, "result": {"versions": {}}}
        elif self.responder is not None:
            reply = {"ok": True, "result": self.responder(method, params)}
        else:
            raise AssertionError(f"Unexpected RPC: {method}")
        if self.deliver_reply is None:
            raise AssertionError("Transport has not been attached")
        self.deliver_reply({**reply, "id": payload["id"]})

    def close(self) -> None:
        self.is_open = False


@dataclass
class MeasureClient:
    context: MeasureToolContext
    transport: WireTransport
    tools: ToolTable

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.tools[name]["handler"](arguments)

    def observe_versions(self, versions: dict[str, int]) -> None:
        self.transport.replies["resources.versions"] = {
            "ok": True,
            "result": {"versions": versions},
        }
        self.transport.replies["state.has_soc"] = {
            "ok": True,
            "result": {"value": False},
        }
        self.context.send_gui_rpc("state.has_soc", {})
        self.transport.sent.clear()


def make_client(tmp_path: Path, responder: RpcResponder | None = None) -> MeasureClient:
    config = MCPBridgeConfig(
        tool_prefix="gui_",
        server_display_name="measure-test",
        server_instructions="",
        app_name="gui",
        default_port=8765,
        mcp_version=74,
        wire_version=55,
        pid_file=tmp_path / "unused.pid",
        log_file=tmp_path / "unused.log",
        run_script_name="unused.py",
    )
    session = MeasureMcpSession(
        config,
        resolve_connect_port=lambda config, requested: config.default_port,
        port_is_open=lambda port: False,
    )
    bridge = McpBridge(config, on_event=session.deliver_event)
    session.attach_bridge(bridge)
    transport = WireTransport(responder)
    bridge.set_transport(transport)
    context = MeasureToolContext(
        config,
        session,
        METHOD_SPECS,
        resolve_connect_port=lambda config, requested: config.default_port,
    )
    return MeasureClient(context, transport, build_measure_tools(context))
