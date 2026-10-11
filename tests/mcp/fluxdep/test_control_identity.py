"""MCP identity tools composed with the real Fluxdep application dispatch."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.gui.app.fluxdep.remote.wire_version import GUI_VERSION, WIRE_VERSION
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.mcp.core.bridge import McpBridge, MCPBridgeConfig
from zcu_tools.mcp.fluxdep.assembly import FluxdepServer, build_fluxdep_server

from tests.gui.app.fluxdep.remote._route_harness import RouteHarness


class RouteTransport:
    """Transport replaces framing only, preserving real validation and authority."""

    def __init__(self, harness: RouteHarness) -> None:
        self.harness = harness
        self.link = ClientLink("mcp", token_required=False)
        harness.service.on_client_open(self.link)
        harness.links.append(self.link)
        self.is_open = True
        self.deliver_reply = None
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def attach(self, deliver_reply, deliver_event, on_closed) -> None:
        self.deliver_reply = deliver_reply

    def send_line(self, payload: dict[str, Any]) -> None:
        self.sent.append((payload["method"], payload["params"]))
        if payload["method"] == "wire.version":
            reply = {
                "ok": True,
                "result": {"wire_version": WIRE_VERSION, "gui_version": GUI_VERSION},
            }
        else:
            reply = self.harness.request(
                self.link, payload["method"], **payload["params"]
            )
        self.deliver_reply({**reply, "id": payload["id"]})

    def close(self) -> None:
        if self.is_open:
            self.harness.disconnect(self.link)
        self.is_open = False


class RouteBridge(McpBridge):
    """Open a fresh application link instead of a socket; no identity defaults."""

    def __init__(self, config: MCPBridgeConfig, harness: RouteHarness) -> None:
        super().__init__(config)
        self.harness = harness
        self.transports: list[RouteTransport] = []

    def connect(self, port: int, token: str | None = None) -> str:
        self.disconnect()
        transport = RouteTransport(self.harness)
        self.transports.append(transport)
        self.set_transport(transport)
        return "attached"


@pytest.fixture
def control_server(
    tmp_path: Path,
) -> Iterator[tuple[FluxdepServer, RouteHarness, RouteBridge]]:
    harness = RouteHarness.create(str(tmp_path))
    config = MCPBridgeConfig(
        app_name="fluxdep",
        tool_prefix="fluxdep_",
        default_port=8766,
        mcp_version=2,
        wire_version=WIRE_VERSION,
        server_display_name="Fluxdep test",
        server_instructions="",
        pid_file=tmp_path / "gui.pid",
        log_file=tmp_path / "gui.log",
        run_script_name="run_fluxdep_gui.py",
    )
    bridge = RouteBridge(config, harness)
    server = build_fluxdep_server(config, tmp_path, bridge=bridge)
    try:
        yield server, harness, bridge
    finally:
        bridge.disconnect()
        harness.close()


def invoke(server: FluxdepServer, name: str, **arguments):
    """Exercise the assembled public tool handler."""
    return server.tools["fluxdep_" + name]["handler"](arguments)


def test_owner_connect_acquires_reports_control_and_exposes_release_acquire(
    control_server,
):
    server, harness, bridge = control_server
    reply = invoke(server, "connect", label="sr-b4")
    expected = {"holder": "rpc:sr-b4", "locked": False, "online": True, "me": True}
    assert isinstance(reply, dict), reply
    assert reply["control"] == expected
    assert bridge.transports[-1].sent[:2] == [
        ("wire.version", {}),
        ("control.connect", {"label": "sr-b4", "mode": "owner"}),
    ]
    assert invoke(server, "control_status") == expected
    assert invoke(server, "control_release")["holder"] == "gui"
    assert invoke(server, "control_acquire")["me"] is True


def test_locked_owner_connect_succeeds_but_acquire_preserves_native_failure(
    control_server,
):
    server, harness, bridge = control_server
    harness.ctrl.control_identity.reclaim_local()
    reply = invoke(server, "connect", label="sr-b4")
    assert isinstance(reply, dict), reply
    assert reply["control"] == {
        "holder": "gui",
        "locked": True,
        "online": True,
        "me": False,
    }
    with pytest.raises(RuntimeError) as caught:
        invoke(server, "control_acquire")
    assert caught.value.reason == "control_locked"
    assert caught.value.data == {"holder": "gui", "locked": True, "online": True}


def test_readonly_connect_reads_rejects_writes_and_reconnects_with_same_identity(
    control_server,
):
    server, harness, bridge = control_server
    reply = invoke(server, "connect", label="reader", mode="readonly")
    assert isinstance(reply, dict), reply
    assert reply["control"]["me"] is False
    original = invoke(server, "project_info")
    with pytest.raises(RuntimeError) as caught:
        invoke(server, "project_setup", chip_name="changed", qub_name="q")
    assert caught.value.reason == "not_holder"
    assert caught.value.data == {"holder": "gui", "locked": False, "online": True}
    assert invoke(server, "project_info") == original
    with pytest.raises(RuntimeError) as caught:
        invoke(server, "control_acquire")
    assert caught.value.reason == "readonly_connection"
    invoke(server, "disconnect")
    reply = invoke(server, "connect")
    assert reply["control"]["me"] is False
    assert bridge.transports[-1].sent[1] == (
        "control.connect",
        {"label": "reader", "mode": "readonly"},
    )


def test_other_online_holder_surfaces_native_control_held(control_server):
    server, harness, bridge = control_server
    harness.client()
    reply = invoke(server, "connect", label="other")
    assert isinstance(reply, dict), reply
    assert reply["control"]["holder"] == "rpc:agent"
    with pytest.raises(RuntimeError) as caught:
        invoke(server, "control_acquire")
    assert caught.value.reason == "control_held"
    assert caught.value.data == {"holder": "rpc:agent", "locked": False, "online": True}


@pytest.mark.parametrize(
    "reason", ["not_holder", "control_locked", "control_held", "readonly_connection"]
)
def test_stdio_projects_real_control_rejections_without_losing_holder_data(
    control_server, reason
):
    from tests.mcp._stdio_support import invoke_stdio

    server, harness, bridge = control_server
    mode = "owner"
    holder = "gui"
    locked = reason == "control_locked"
    method = "control_acquire"
    if locked:
        harness.ctrl.control_identity.reclaim_local()
    elif reason == "control_held":
        harness.client()
        holder = "rpc:agent"
    else:
        mode = "readonly"
    invoke(server, "connect", label="reader", mode=mode)
    if reason == "not_holder":
        method = "project_setup"
    arguments = (
        {"chip_name": "changed", "qub_name": "q"} if reason == "not_holder" else {}
    )
    reply = invoke_stdio(
        server.bridge.config, server.tools, "fluxdep_" + method, arguments
    )
    assert reply["result"]["isError"] is True
    assert reply["result"]["content"][0]["text"].startswith("{"), reply
    error = json.loads(reply["result"]["content"][0]["text"])["error"]
    assert error["code"] == "precondition_failed"
    assert error["reason"] == reason
    assert error["data"] == {"holder": holder, "locked": locked, "online": True}
