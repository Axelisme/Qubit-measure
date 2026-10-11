"""Identity initialization and native control diagnostics at public MCP seams."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from zcu_tools.gui.remote.control_methods import CONTROL_METHOD_SPECS
from zcu_tools.gui.remote.param_spec import build_input_schema
from zcu_tools.mcp.core.bridge import McpBridge
from zcu_tools.mcp.core.rpc import GuiInvocationError
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import GuiRpcError, MeasureMcpSession
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
    catalog_reply = transport.replies["rpc.catalog"]
    if callable(catalog_reply):
        raise RuntimeError("identity fixture requires a catalog response")
    catalog = catalog_reply["result"]["methods"]
    catalog.extend(
        {
            "method": method,
            "description": spec.description,
            "params": build_input_schema(spec.params),
            "timeout_seconds": spec.timeout_seconds,
            "exposure": "rpc",
            "tool_names": [],
            "operation_key": None,
        }
        for method, spec in CONTROL_METHOD_SPECS.items()
    )
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
    with pytest.raises(GuiRpcError) as caught:
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


@pytest.mark.parametrize(
    "reason", ["not_holder", "control_locked", "control_held", "readonly_connection"]
)
def test_stdio_preserves_control_error_envelope(identity_client, reason):
    from tests.mcp._stdio_support import invoke_stdio

    client = identity_client
    projection = {"holder": "gui", "locked": True, "online": True, "me": False}
    client.transport.replies["control.connect"] = {"ok": True, "result": projection}
    client.call("connect", {"mode": "readonly"})
    error = {
        "code": "precondition_failed",
        "message": "native diagnostic",
        "reason": reason,
        "data": {"holder": "gui", "locked": True, "online": True},
    }
    client.transport.replies["control.acquire"] = {"ok": False, "error": error}
    reply = invoke_stdio(
        client.context.config, client.tools, "rpc_call", {"method": "control.acquire"}
    )
    assert reply["result"]["isError"] is True
    assert reply["result"]["content"][0]["text"].startswith("{"), reply
    assert json.loads(reply["result"]["content"][0]["text"]) == {"error": error}


def test_auth_rejection_before_catalog_keeps_existing_auth_reason(identity_client):
    client = identity_client
    client.transport.replies["control.connect"] = {
        "ok": False,
        "error": {"code": "unauthorized", "message": "token required"},
    }
    with pytest.raises(GuiInvocationError) as caught:
        client.call("connect", {})
    assert caught.value.reason == "unauthorized"
    assert not client.context.bridge.is_connected
    assert all(method != "rpc.catalog" for method, _ in client.transport.sent)
