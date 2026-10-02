"""Connection-bound calls cannot cross GUI incarnations or overlap RPC state."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from zcu_tools.mcp.measure import tools_operation
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import GuiRpcError

from ._support import MeasureClient, WireTransport, make_client


def make_restartable_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[MeasureClient, WireTransport]:
    client = make_client(tmp_path, port_is_open=lambda port: True)
    second = WireTransport()
    second.replies["rpc.catalog"] = deepcopy(client.transport.replies["rpc.catalog"])

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(second)
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    return client, second


def test_binding_survives_noop_connect_but_not_a_new_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    session = client.context.session
    first = session.bind()
    client.transport.replies["device.connect"] = {
        "ok": True,
        "result": {"operation_id": 1},
    }
    second.replies["device.connect"] = client.transport.replies["device.connect"]
    old_handle = first.send_gui_rpc("device.connect", {"name": "flux"})["handle"]
    session.connect_to_gui(port=None, launch="never", clean=False)
    assert first.expose_operation(1) == old_handle
    assert first.catalog["device.connect"]["method"] == "device.connect"

    session.connect_to_gui(port=9912, launch="never", clean=False)
    sent = list(second.sent)
    for call in (
        lambda: first.send_gui_rpc("device.connect", {"name": "flux"}),
        lambda: first.read_internal("operation.active", {}),
        lambda: first.expose_operation(1),
        lambda: first.catalog,
    ):
        with pytest.raises(GuiRpcError, match="expired") as error:
            call()
        assert error.value.reason == "connection_lost"
    assert second.sent == sent
    current = session.bind()
    new_handle = current.send_gui_rpc("device.connect", {"name": "flux"})["handle"]
    assert new_handle != old_handle
    with pytest.raises(GuiRpcError) as error:
        current.read_internal("operation.cancel", {}, operation_handle=old_handle)
    assert error.value.reason == "unknown_op"


def test_delivered_reply_remains_known_when_its_socket_then_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    binding = client.context.session.bind()
    client.transport.replies["context.labels"] = {
        "ok": True,
        "result": {"labels": ["original-gui"]},
    }
    real_send = client.transport.send_line

    def send(payload: dict[str, Any]) -> None:
        real_send(payload)
        client.context.bridge.disconnect()

    monkeypatch.setattr(client.transport, "send_line", send)
    assert binding.send_gui_rpc("context.labels", {}) == {"labels": ["original-gui"]}
    with pytest.raises(GuiRpcError) as error:
        binding.read_internal("operation.active", {})
    assert error.value.reason == "connection_lost"
    assert not second.sent


def test_pending_eof_reports_connection_lost_without_replaying_rpc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    binding = client.context.session.bind()
    sent = Event()
    client.transport.sent.clear()

    def send(payload: dict[str, Any]) -> None:
        client.transport.sent.append((payload["method"], payload["params"]))
        sent.set()

    monkeypatch.setattr(client.transport, "send_line", send)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(binding.send_gui_rpc, "context.labels", {}, timeout=1.0)
        try:
            assert sent.wait(1)
            client.transport.close()
            assert client.transport.on_closed is not None
            client.transport.on_closed(None)
            with pytest.raises(GuiRpcError) as error:
                pending.result(timeout=1)
            assert error.value.reason == "connection_lost"
            assert isinstance(error.value.__context__, ConnectionError)
        finally:
            client.context.bridge.disconnect()

    assert client.transport.sent == [("context.labels", {})]
    assert not second.sent


def test_catalog_and_operation_snapshots_cannot_modify_session_state(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    session = client.context.session
    bound = session.bind()
    client.transport.replies["device.connect"] = {
        "ok": True,
        "result": {"operation_id": 7},
    }
    handle = bound.send_gui_rpc("device.connect", {"name": "flux"})["handle"]
    expected = bound.catalog
    for catalog in (session.catalog, bound.catalog):
        catalog["device.connect"]["tool_names"].append("test-alias")
        catalog["device.connect"]["params"]["test-alias"] = True
        assert bound.catalog == expected
    handles = session.operation_handles
    assert handle in handles.values()
    with pytest.raises(TypeError):
        handles["replacement"] = handle  # type: ignore[index]
    assert "replacement" not in session.debug_operations()["handles"]


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("status", {}),
        ("wait", {"op": 1, "timeout": 0}),
        ("cancel", {"op": 1}),
        ("tab_live", {"tab": "t"}),
        ("rpc_call", {"method": "adapter.list"}),
        ("rpc_describe", {"method": "adapter.list"}),
        ("rpc_list", {}),
        ("tab_get", {"tab": "t"}),
    ],
)
def test_assembled_tools_keep_an_inherited_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    arguments: dict[str, Any],
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    tools = build_measure_tools(client.context.bound())
    client.context.session.connect_to_gui(port=9912, launch="never", clean=False)
    sent = list(second.sent)
    with pytest.raises(GuiRpcError) as error:
        tools[tool]["handler"](arguments)
    assert error.value.reason == "connection_lost"
    assert second.sent == sent


@pytest.mark.parametrize("tool", ["status", "tab_live"])
def test_operation_discovery_does_not_expose_an_old_reply_in_a_new_gui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    client.transport.replies.update(
        {
            "state.has_project": {"ok": True, "result": {"value": False}},
            "state.has_active_context": {"ok": True, "result": {"value": False}},
            "state.has_soc": {"ok": True, "result": {"value": False}},
            "context.active": {"ok": True, "result": {"label": None}},
            "device.list": {"ok": True, "result": {"devices": []}},
            "predictor.info": {"ok": True, "result": {"loaded": False}},
            "tab.snapshot": {
                "ok": True,
                "result": {
                    "tabs": [
                        {
                            "tab_id": "t",
                            "adapter_name": "lookback",
                            "interaction": {
                                "is_running": True,
                                "has_run_result": False,
                            },
                        }
                    ]
                },
            },
        }
    )

    client.transport.replies["operation.active"] = {
        "ok": True,
        "result": {"operations": [{"op": 1, "kind": "run", "tab": "t"}]},
    }
    real_send = client.context.bridge.send_rpc_raw

    def reconnect_after_reply(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        reply = real_send(method, params, timeout_seconds)
        if method == "operation.active":
            client.context.session.connect_to_gui(
                port=9912, launch="never", clean=False
            )
        return reply

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", reconnect_after_reply)
    with pytest.raises(GuiRpcError) as error:
        client.call(tool, {"tab": "t"})
    assert error.value.reason == "connection_lost"
    assert not any(method == "operation.progress" for method, _ in second.sent)
    assert client.context.session.bind().expose_operation(9) == 1


def test_wait_keeps_its_binding_between_await_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    handle = client.context.session.bind().expose_operation(1)
    client.transport.replies["operation.await"] = {
        "ok": True,
        "result": {"reason": "timeout"},
    }
    ticks = iter((0.0, 1.0))

    def clock() -> float:
        value = next(ticks)
        if value:
            client.context.session.connect_to_gui(
                port=9912, launch="never", clean=False
            )
        return value

    monkeypatch.setattr(tools_operation, "time", SimpleNamespace(monotonic=clock))
    with pytest.raises(GuiRpcError) as error:
        client.call("wait", {"op": handle, "timeout": 0})
    assert error.value.reason == "connection_lost"
    assert not any(method == "operation.progress" for method, _ in second.sent)


def make_serialization_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[MeasureClient, WireTransport]:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    catalog = [
        {
            "method": "test.start",
            "description": "Start one operation",
            "params": {"type": "object"},
            "timeout_seconds": 5.0,
            "exposure": "rpc",
            "tool_names": [],
            "operation_key": "job:old",
        }
    ]
    client.transport.replies["rpc.catalog"] = {
        "ok": True,
        "result": {"methods": catalog},
    }
    second.replies["rpc.catalog"] = {
        "ok": True,
        "result": {
            "methods": [
                {**catalog[0], "timeout_seconds": 9.0, "operation_key": "job:new"}
            ]
        },
    }
    second.replies["test.start"] = {"ok": True, "result": {"operation_id": 1}}
    return client, second


@pytest.mark.parametrize("action", ["rpc", "connect"])
def test_rpc_and_connect_serialize_catalog_timeout_and_handle_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    client, _ = make_serialization_client(tmp_path, monkeypatch)
    entered, release, attempted, finished = Event(), Event(), Event(), Event()
    calls = 0

    def reply(params: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(3), "test did not release the first RPC"
        return {"ok": True, "result": {"operation_id": calls}}

    client.transport.replies["test.start"] = reply
    binding = client.context.session.bind()
    observed_timeouts: list[float] = []
    real_send = client.context.bridge.send_rpc_raw

    def send(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        if method == "test.start":
            observed_timeouts.append(timeout_seconds)
        return real_send(method, params, timeout_seconds)

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", send)

    def competing_call() -> dict[str, Any]:
        attempted.set()
        try:
            if action == "rpc":
                return binding.send_gui_rpc("test.start", {})
            return client.context.session.connect_to_gui(
                port=9912, launch="never", clean=False
            )
        finally:
            finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(binding.send_gui_rpc, "test.start", {})
        try:
            assert entered.wait(3)
            other = pool.submit(competing_call)
            assert attempted.wait(3)
            assert not finished.wait(0.1), "competing call crossed the active RPC"
        finally:
            release.set()
        old_result = first.result(timeout=3)
        other_result = other.result(timeout=3)
    assert old_result["handle"] == 1
    if action == "rpc":
        assert other_result["handle"] == 2
        assert observed_timeouts == [6.0, 6.0]
        assert client.context.session.operation_handles == {"job:old": 2}
    else:
        new = client.context.session.bind().send_gui_rpc("test.start", {})
        assert new["handle"] == 2
        assert observed_timeouts == [6.0, 10.0]
        assert client.context.session.operation_handles == {"job:new": 2}
        with pytest.raises(GuiRpcError) as error:
            binding.send_gui_rpc("test.start", {})
        assert error.value.reason == "connection_lost"
