"""Measure MCP connection and live GUI catalog contracts."""

from pathlib import Path
from typing import Any

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError

from ._support import make_client

CATALOG = [
    {
        "method": "adapter.list",
        "description": "List adapters",
        "params": {"type": "object", "properties": {}},
        "timeout_seconds": 5.0,
        "exposure": "rpc",
        "tool_names": [],
        "guard_deps": [],
        "reveals": [],
        "operation_key": None,
    },
    {
        "method": "adapter.guide",
        "description": "Read a guide",
        "params": {
            "type": "object",
            "properties": {"adapter_name": {"type": "string"}},
            "required": ["adapter_name"],
        },
        "timeout_seconds": 5.0,
        "exposure": "tool",
        "tool_names": ["guide"],
        "guard_deps": [],
        "reveals": [],
        "operation_key": None,
    },
]


def overview_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    replies: dict[str, dict[str, Any]] = {
        "state.has_project": {"value": False},
        "state.has_context": {"value": False},
        "state.has_active_context": {"value": False},
        "state.has_soc": {"value": False},
        "tab.snapshot": {"tabs": []},
        "context.active": {"label": None},
        "device.list": {"devices": []},
        "predictor.info": {"loaded": False},
        "operation.active": {"operations": []},
        "state.hardware_gate": {},
        "run.running_tab": {"tab_id": None},
        "view.snapshot": {"active_tab_id": None},
    }
    return replies[method]


def test_connect_loads_catalog_and_rpc_tools_route_by_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    client.transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {"methods": CATALOG},
    }
    client.transport.replies["adapter.list"] = {
        "ok": True,
        "result": {"adapters": ["fake"]},
    }

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(client.transport)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    result = client.call("connect", {"port": 9912})
    assert result["port"] == 9912
    assert result["launched"] is False
    assert result["versions"]["wire"] == client.context.config.wire_version
    assert result["status"]
    assert client.transport.sent[:2] == [("wire.version", {}), ("rpc.catalog", {})]
    assert client.call("rpc_list", {"domain": "adapter"})["methods"] == [
        {"method": "adapter.list", "description": "List adapters", "tool_names": []},
        {
            "method": "adapter.guide",
            "description": "Read a guide",
            "tool_names": ["guide"],
        },
    ]
    assert client.call("rpc_describe", {"method": "adapter.list"})["params"] == {
        "type": "object",
        "properties": {},
    }
    assert client.call("rpc_call", {"method": "adapter.list", "params": {}}) == {
        "adapters": ["fake"]
    }
    with pytest.raises(RuntimeError) as error:
        client.call(
            "rpc_call", {"method": "adapter.guide", "params": {"adapter_name": "fake"}}
        )
    assert getattr(error.value, "reason", None) == "use_tool"
    with pytest.raises(RuntimeError) as error:
        client.call("rpc_describe", {"method": "rpc.catalog"})
    assert getattr(error.value, "reason", None) == "unknown_method"
    assert all(
        method not in {"adapter.guide", "rpc.catalog"}
        for method, _ in client.transport.sent[2:]
    )


def test_incompatible_wire_refuses_catalog_and_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    client.transport.replies["wire.version"] = {
        "ok": True,
        "result": {"wire_version": -1, "gui_version": 100},
    }

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(client.transport)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    with pytest.raises(RuntimeError, match="wire"):
        client.call("connect", {"port": 9912})
    assert client.transport.sent == [("wire.version", {})]
    assert not client.context.bridge.is_connected


def test_reconnect_replaces_catalog_without_replaying_previous_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    first = client.transport
    first.replies["rpc.catalog"] = {"ok": True, "result": {"methods": CATALOG}}
    first.replies["adapter.list"] = {"ok": True, "result": {"adapters": ["old"]}}
    client.context.bridge.set_transport(None)
    next_transport = type(first)(overview_rpc)
    next_transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {
            "methods": [{**CATALOG[0], "method": "project.info"}],
        },
    }
    next_transport.replies["project.info"] = {"ok": True, "result": {"chip": "new"}}
    transports = iter((first, next_transport))
    ports: list[int] = []

    def connect(port: int, token: str | None = None) -> str:
        ports.append(port)
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    client.call("connect", {"port": 9912})
    assert client.call("rpc_call", {"method": "adapter.list", "params": {}}) == {
        "adapters": ["old"]
    }
    client.context.bridge.disconnect()
    assert client.call("rpc_call", {"method": "project.info", "params": {}}) == {
        "chip": "new"
    }
    with pytest.raises(RuntimeError) as error:
        client.call("rpc_call", {"method": "adapter.list"})
    assert getattr(error.value, "reason", None) == "unknown_method"
    assert [method for method, _ in first.sent].count("adapter.list") == 1
    assert all(method != "adapter.list" for method, _ in next_transport.sent)
    assert ports == [9912, 9912]


def test_discovered_gui_can_restart_on_a_different_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered = iter((9911, 9913))
    client = make_client(
        tmp_path,
        overview_rpc,
        resolve_connect_port=lambda config, requested: (
            requested if requested is not None else next(discovered)
        ),
        port_is_open=lambda port: True,
    )
    client.context.bridge.set_transport(None)
    first = client.transport
    first.replies["rpc.catalog"] = {"ok": True, "result": {"methods": CATALOG}}
    second = type(first)(overview_rpc)
    second.replies["rpc.catalog"] = {"ok": True, "result": {"methods": CATALOG}}
    second.replies["adapter.list"] = {"ok": True, "result": {"adapters": ["new"]}}
    transports = iter((first, second))
    ports: list[int] = []

    def connect(port: int, token: str | None = None) -> str:
        ports.append(port)
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    assert client.call("connect", {})["port"] == 9911
    client.context.bridge.disconnect()
    assert client.call("rpc_call", {"method": "adapter.list"}) == {
        "adapters": ["new"],
    }
    assert ports == [9911, 9913]


def test_ambiguous_mutation_timeout_does_not_replay_on_reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    first = client.transport
    second = type(first)(overview_rpc)
    second.replies["rpc.catalog"] = {
        "ok": True,
        "result": {
            "methods": [
                {**CATALOG[0], "method": "project.info"},
            ]
        },
    }
    second.replies["project.info"] = {"ok": True, "result": {"chip": "new"}}
    client.context.session.ensure_connected()
    real_send = client.context.bridge.send_rpc_raw
    attempted: list[str] = []

    def send(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        if method == "adapter.list":
            attempted.append(method)
            client.context.bridge.disconnect()
            raise GuiTransportTimeoutError(method, timeout_seconds)
        return real_send(method, params, timeout_seconds)

    def reconnect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(second)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", send)
    monkeypatch.setattr(client.context.bridge, "connect", reconnect)
    with pytest.raises(RuntimeError) as error:
        client.call("rpc_call", {"method": "adapter.list", "params": {}})
    assert getattr(error.value, "reason", None) == "gui_transport_timeout"
    assert client.call("rpc_call", {"method": "project.info"}) == {"chip": "new"}
    assert attempted == ["adapter.list"]
    assert all(method != "adapter.list" for method, _ in second.sent)


def test_connect_switches_an_explicit_port_and_discards_previous_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    first = client.transport
    first.replies["rpc.catalog"] = {"ok": True, "result": {"methods": CATALOG}}
    other = type(first)(overview_rpc)
    other.replies["rpc.catalog"] = {
        "ok": True,
        "result": {
            "methods": [
                {**CATALOG[0], "method": "project.info"},
            ]
        },
    }
    ports: list[int] = []
    transports = iter((first, other))

    def connect(port: int, token: str | None = None) -> str:
        ports.append(port)
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    client.call("connect", {"port": 9911})
    client.context.session.last_seen_versions["context"] = 12
    client.context.session.operation_handles["tab:old"] = 43
    client.call("connect", {"port": 9912})
    assert ports == [9911, 9912]
    assert not client.context.session.operation_handles
    assert "context" not in client.context.session.last_seen_versions
    assert client.call("rpc_list", {})["methods"] == [
        {"method": "project.info", "description": "List adapters", "tool_names": []},
    ]


def test_connect_launch_modes_do_not_start_hardware_or_kill_an_existing_gui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client(tmp_path, overview_rpc, port_is_open=lambda port: False)
    client.context.bridge.set_transport(None)
    client.transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {"methods": CATALOG},
    }
    calls: list[tuple[int, list[str] | None]] = []

    def launch(
        repo_root: Path,
        port: int,
        token: str | None = None,
        auto_connect: bool = True,
        extra_args: list[str] | None = None,
    ) -> str:
        calls.append((port, extra_args))
        client.context.bridge.set_transport(client.transport)
        return "launched"

    monkeypatch.setattr(client.context.bridge, "launch", launch)
    with pytest.raises(RuntimeError) as error:
        client.call("connect", {})
    assert getattr(error.value, "reason", None) == "no_gui"
    assert calls == []
    result = client.call("connect", {"launch": "if_missing", "clean": True})
    assert result["launched"] is True
    assert calls == [(8765, ["--clean"])]
    assert all(method != "soc.connect" for method, _ in client.transport.sent)

    busy = make_client(tmp_path, port_is_open=lambda port: True)
    busy.context.bridge.set_transport(None)
    with pytest.raises(RuntimeError) as error:
        busy.call("connect", {"launch": "new"})
    assert getattr(error.value, "reason", None) == "port_in_use"


@pytest.mark.parametrize(
    "catalog", [{"methods": "invalid"}, {"methods": [CATALOG[0], CATALOG[0]]}]
)
def test_connect_rejects_malformed_or_duplicate_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    catalog: dict[str, Any],
) -> None:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    client.context.bridge.set_transport(None)
    client.transport.replies["rpc.catalog"] = {"ok": True, "result": catalog}

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(client.transport)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    with pytest.raises(RuntimeError, match="catalog"):
        client.call("connect", {"port": 9912})
    assert not client.context.bridge.is_connected
