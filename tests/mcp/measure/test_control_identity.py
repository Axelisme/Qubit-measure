"""Identity initialization and native control diagnostics at public MCP seams."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import MeasureMcpSession
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

from ._support import MeasureClient, WireTransport, make_client


class AttachedBridge(McpBridge):
    """Replace only socket opening; handshake and tools still execute normally."""

    def __init__(self, config, transport: WireTransport) -> None:
        super().__init__(config)
        self.transport = transport
        self.connects = 0

    def connect(self, port: int, token: str | None = None) -> str:
        self.disconnect()
        self.connects += 1
        self.transport.is_open = True
        self.set_transport(self.transport)
        return "attached"


def orientation(method: str, params: dict[str, Any]) -> dict[str, Any]:
    replies = {
        "state.has_project": {"value": False},
        "state.has_active_context": {"value": False},
        "state.has_soc": {"value": False},
        "tab.snapshot": {"tabs": []},
        "context.active": {"label": None},
        "device.list": {"devices": []},
        "predictor.info": {"loaded": False},
        "operation.active": {"operations": []},
        "entry.info": {"entry": None},
    }
    return replies[method]


@pytest.fixture
def identity_client(tmp_path: Path) -> Iterator[MeasureClient]:
    seed = make_client(tmp_path, recipes=[])
    config = seed.context.config
    seed.context.session.close()
    transport = seed.transport
    transport.responder = orientation
    bridge = AttachedBridge(config, transport)
    session = MeasureMcpSession(
        config,
        recipes=[],
        bridge=bridge,
        resolve_connect_port=lambda config, port: port or config.default_port,
        port_is_open=lambda port: True,
    )
    context = MeasureToolContext(
        config, session, resolve_connect_port=lambda c, p: p or c.default_port
    )
    client = MeasureClient(context, transport, build_measure_tools(context, recipes=[]))
    try:
        yield client
    finally:
        session.close()


@pytest.mark.parametrize(
    ("arguments", "projection"),
    [
        ({}, {"holder": "rpc:agent", "locked": False, "online": True, "me": True}),
        (
            {"label": "sr-b4"},
            {"holder": "gui", "locked": True, "online": True, "me": False},
        ),
        (
            {"label": "reader", "mode": "readonly"},
            {"holder": "gui", "locked": False, "online": True, "me": False},
        ),
    ],
)
def test_connect_initializes_before_catalog_and_reports_native_control(
    identity_client, arguments, projection
):
    client = identity_client
    client.transport.replies["control.connect"] = {"ok": True, "result": projection}
    client.transport.replies["control.status"] = {"ok": True, "result": projection}
    reply = client.call("connect", arguments)
    assert reply.get("control") == projection
    assert client.transport.sent[:3] == [
        ("wire.version", {}),
        (
            "control.connect",
            {
                "label": arguments.get("label", "agent"),
                "mode": arguments.get("mode", "owner"),
            },
        ),
        ("rpc.catalog", {}),
    ]
    assert client.call("rpc_call", {"method": "control.status"}) == projection
    assert client.call("rpc_call", {"method": "entry.info"}) == {"entry": None}
    client.context.bridge.disconnect()
    client.call("rpc_call", {"method": "entry.info"})
    initializations = [
        params
        for method, params in client.transport.sent
        if method == "control.connect"
    ]
    assert initializations == [initializations[0], initializations[0]]


def test_reconnect_retains_identity_and_repeat_connect_queries_current_status(
    identity_client,
):
    client = identity_client
    remote = {"holder": "rpc:sr-b4", "locked": False, "online": True, "me": True}
    local = {"holder": "gui", "locked": False, "online": True, "me": False}
    client.transport.replies["control.connect"] = {"ok": True, "result": remote}
    client.transport.replies["control.status"] = {"ok": True, "result": remote}
    assert (
        client.call("connect", {"label": "sr-b4", "mode": "owner"}).get("control")
        == remote
    )
    client.transport.replies["control.release"] = {"ok": True, "result": local}
    assert client.call("rpc_call", {"method": "control.release"}) == local
    client.transport.replies["control.status"] = {"ok": True, "result": local}
    assert client.call("connect", {}).get("control") == local
    client.context.bridge.disconnect()
    client.call("rpc_call", {"method": "entry.info"})
    assert [
        params
        for method, params in client.transport.sent
        if method == "control.connect"
    ] == [
        {"label": "sr-b4", "mode": "owner"},
        {"label": "sr-b4", "mode": "owner"},
    ]


@pytest.mark.parametrize(
    "reason", ["not_holder", "control_locked", "control_held", "readonly_connection"]
)
def test_control_errors_retain_native_code_reason_and_holder_data(
    identity_client, reason
):
    client = identity_client
    projection = {"holder": "gui", "locked": True, "online": True, "me": False}
    client.transport.replies["control.connect"] = {"ok": True, "result": projection}
    client.transport.replies["control.status"] = {"ok": True, "result": projection}
    client.call("connect", {"mode": "readonly"})
    method = "tab.close" if reason == "not_holder" else "control.acquire"
    error = {
        "code": "precondition_failed",
        "message": "native diagnostic",
        "reason": reason,
        "data": {"holder": "gui", "locked": True, "online": True},
    }
    client.transport.replies[method] = {"ok": False, "error": error}
    with pytest.raises(RuntimeError) as caught:
        client.call(
            "rpc_call",
            {
                "method": method,
                "params": {"tab_id": "missing"} if method == "tab.close" else {},
            },
        )
    assert getattr(caught.value, "data", None) == error["data"]
    assert caught.value.code == error["code"]
    assert caught.value.reason == reason
