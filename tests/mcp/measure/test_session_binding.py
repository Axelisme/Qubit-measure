"""Connection-bound calls cannot cross GUI incarnations or overlap RPC state."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from zcu_tools.mcp.measure import tools_operation
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import GuiConnection, GuiRpcError

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


def prime_operation_discovery(client: MeasureClient, gui_id: int = 1) -> None:
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
        "result": {"operations": [{"op": gui_id, "kind": "run", "tab": "t"}]},
    }


def test_status_received_discovery_reply_survives_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    binding = client.context.session.bind()
    prime_operation_discovery(client, gui_id=77)
    real_send = client.transport.send_line

    def send(payload: dict[str, Any]) -> None:
        real_send(payload)
        if payload["method"] == "operation.active":
            client.context.bridge.disconnect()

    monkeypatch.setattr(client.transport, "send_line", send)
    result = client.call("status", {})
    assert result["tabs"] == [{"tab": "t", "experiment": "lookback", "running": True}]
    assert result["running"] == [{"op": 1, "kind": "run", "tab": "t"}]
    assert result["context"] == {"active": None}
    with pytest.raises(GuiRpcError) as error:
        binding.read_internal("operation.active", {})
    assert error.value.reason == "connection_lost"
    assert not second.sent
    assert client.transport.sent.count(("operation.active", {})) == 1
    client.context.session.connect_to_gui(port=9912, launch="never", clean=False)
    assert client.context.session.bind().expose_operation(77) == 2


@pytest.mark.parametrize("connection_change", ["eof", "reconnect"])
@pytest.mark.parametrize(
    ("tool", "arguments", "replies", "cut_after"),
    [
        pytest.param(
            "experiments",
            {},
            {"adapter.list": {"adapters": ["lookback"]}},
            "adapter.list",
            id="experiment-guides",
        ),
        pytest.param(
            "tab_open",
            {"experiment": "lookback"},
            {
                "tab.list_all": {"active_tab_id": "previous"},
                "tab.new": {"tab_id": "created"},
            },
            "tab.new",
            id="tab-activation-and-cleanup",
        ),
        pytest.param(
            "tab_get",
            {"tab": "t", "include": ["cfg", "analyze_params"]},
            {"tab.get_cfg": {"cfg": {}}},
            "tab.get_cfg",
            id="tab-sections",
        ),
        pytest.param(
            "tab_analyze",
            {"tab": "t"},
            {
                "tab.analyze": {
                    "operation_id": 7,
                    "interactive": False,
                    "params": {},
                    "invalidated_on_success": [],
                }
            },
            "tab.analyze",
            id="analysis-wait",
        ),
        pytest.param(
            "tab_analyze",
            {"tab": "t"},
            {
                "tab.analyze": {
                    "operation_id": 7,
                    "interactive": True,
                    "params": {},
                    "invalidated_on_success": [],
                }
            },
            "tab.analyze",
            id="analysis-interaction",
        ),
        pytest.param(
            "tab_save",
            {"tab": "t"},
            {"tab.save_artifacts": {"operation_id": 7, "destinations": {}}},
            "tab.save_artifacts",
            id="save-wait",
        ),
        pytest.param(
            "accept",
            {"tab": "t"},
            {"tab.get_analyze_result": {"summary": {"value": 1}}},
            "tab.get_analyze_result",
            id="accept-before-writes",
        ),
        pytest.param(
            "accept",
            {"tab": "t"},
            {
                "tab.get_analyze_result": {"summary": {"value": 1}},
                "tab.get_post_analyze_result": {"summary": {"value": 2}},
                "tab.writeback_preview": {"has_draft": True, "items": [{"id": "x"}]},
                "tab.writeback_write": {"written": [{"id": "x", "after": 2}]},
            },
            "tab.writeback_write",
            id="accept-confirmed-primary",
        ),
        pytest.param(
            "devices",
            {"name": "bias"},
            {
                "device.snapshot": {
                    "snapshot": {
                        "name": "bias",
                        "type_name": "current",
                        "address": "local",
                        "status": "connected",
                        "error": None,
                    }
                }
            },
            "device.snapshot",
            id="device-live-fields",
        ),
        pytest.param(
            "device_connect",
            {"name": "bias", "type": "current", "address": "local"},
            {"device.connect": {"operation_id": 7}},
            "device.connect",
            id="device-connect-wait",
        ),
        pytest.param(
            "device_connect",
            {"name": "bias"},
            {"device.reconnect": {"operation_id": 7}},
            "device.reconnect",
            id="device-reconnect-wait",
        ),
        pytest.param(
            "device_disconnect",
            {"name": "bias"},
            {"device.disconnect": {"operation_id": 7}},
            "device.disconnect",
            id="device-disconnect-wait",
        ),
        pytest.param(
            "device_set",
            {"name": "bias", "values": {"current": 2}},
            {"device.setup_spec": {"fields": [{"name": "current", "settable": True}]}},
            "device.setup_spec",
            id="device-setup",
        ),
        pytest.param(
            "ml_create",
            {"role_id": "drive"},
            {
                "context.ml_list_roles": {
                    "roles": [
                        {
                            "role_id": "drive",
                            "label": "Drive",
                            "item_kind": "module",
                            "default_name": "drive",
                        }
                    ]
                }
            },
            "context.ml_list_roles",
            id="library-create",
        ),
        pytest.param(
            "ml_edit",
            {"name": "drive", "edits": []},
            {"context.ml_get": {"modules": [{"name": "drive"}], "waveforms": []}},
            "context.ml_get",
            id="library-edit",
        ),
        pytest.param(
            "ml_rename",
            {"name": "drive", "new_name": "drive_new"},
            {"context.ml_get": {"modules": [{"name": "drive"}], "waveforms": []}},
            "context.ml_get",
            id="library-rename",
        ),
        pytest.param(
            "ml_delete",
            {"name": "drive"},
            {"context.ml_get": {"modules": [{"name": "drive"}], "waveforms": []}},
            "context.ml_get",
            id="library-delete",
        ),
        pytest.param(
            "contexts",
            {},
            {"context.labels": {"labels": ["original"]}},
            "context.labels",
            id="context-selection",
        ),
        pytest.param(
            "md_get",
            {"keys": ["alpha", "beta"]},
            {"context.md_get_attr": {"value": 1}},
            "context.md_get_attr",
            id="metadict-reads",
        ),
        pytest.param(
            "md_set",
            {"values": {"alpha": 2, "beta": 3}},
            {"context.md_set_attr": {"before": 1, "after": 2}},
            "context.md_set_attr",
            id="metadict-confirmed-prefix",
        ),
        pytest.param(
            "soc_info",
            {},
            {"state.has_soc": {"value": True}},
            "state.has_soc",
            id="soc-details",
        ),
        pytest.param(
            "soc_connect",
            {"address": "localhost", "port": 8888},
            {"soc.connect": {}},
            "soc.connect",
            id="soc-connect-details",
        ),
        pytest.param(
            "predict",
            {"value": 0.2, "transitions": [[0, 1], [1, 2]]},
            {"predictor.predict": {"freq_mhz": 5000}},
            "predictor.predict",
            id="multiple-transitions",
        ),
    ],
)
def test_assembled_multistep_tools_do_not_cross_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    connection_change: str,
    tool: str,
    arguments: dict[str, Any],
    replies: dict[str, dict[str, Any]],
    cut_after: str,
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    client.transport.replies.update(
        {method: {"ok": True, "result": reply} for method, reply in replies.items()}
    )
    real_read = GuiConnection.read_internal
    real_send = GuiConnection.send_gui_rpc
    changed = False
    replacement_at_cut: list[tuple[str, dict[str, Any]]] = []

    def change_connection(method: str) -> None:
        nonlocal changed
        if method != cut_after or changed:
            return
        changed = True
        if connection_change == "eof":
            client.context.bridge.disconnect()
        else:
            client.context.session.connect_to_gui(
                port=9912, launch="never", clean=False
            )
        replacement_at_cut.extend(second.sent)

    def read(
        connection: GuiConnection,
        method: str,
        params: dict[str, Any],
        *,
        operation_handle: int | None = None,
    ) -> dict[str, Any]:
        reply = real_read(connection, method, params, operation_handle=operation_handle)
        change_connection(method)
        return reply

    def send(
        connection: GuiConnection,
        method: str,
        params: dict[str, Any],
        timeout_seconds: float | None = None,
        *,
        operation_handle: int | None = None,
        before_send: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        reply = real_send(
            connection,
            method,
            params,
            timeout_seconds,
            operation_handle=operation_handle,
            before_send=before_send,
        )
        change_connection(method)
        return reply

    # Change the GUI only after the complete public RPC (including handle conversion).
    monkeypatch.setattr(GuiConnection, "read_internal", read)
    monkeypatch.setattr(GuiConnection, "send_gui_rpc", send)
    if tool == "accept":
        result = client.call(tool, arguments)
        assert result["status"] == "failed"
        assert result["error"]["reason"] == "connection_lost"
        assert result["failed_stage"] == "post"
        assert result["failed_stage_may_have_partial_writes"] is False
        if cut_after == "tab.writeback_write":
            assert result["completed"] == [
                {"stage": "primary", "written": [{"id": "x", "after": 2}]}
            ]
            assert result["not_started"] == []
        else:
            assert result["completed"] == []
            assert result["not_started"] == ["primary"]
    elif tool == "tab_analyze" and not replies["tab.analyze"]["interactive"]:
        result = client.call(tool, arguments)
        assert (
            result.is_error,
            result.data["status"],
            result.data["error"]["reason"],
            result.data["op"],
        ) == (True, "failed", "connection_lost", 1)
        client.context.session.close()
    else:
        with pytest.raises(GuiRpcError) as error:
            client.call(tool, arguments)
        assert error.value.reason == (
            "cleanup_failed" if tool == "tab_open" else "connection_lost"
        )
        if tool == "tab_open":
            assert "tab may remain open" in str(error.value)
        if tool == "md_set":
            assert (
                "failed at 'beta'; confirmed prefix: {'alpha': {'before': 1, 'after': 2}}"
                in str(error.value)
            )
    client.context.session.close()
    assert changed
    assert second.sent == replacement_at_cut
    assert sum(method == cut_after for method, _ in client.transport.sent) == 1


@pytest.mark.parametrize("method", ["tab.run_start", "tab.analyze"])
def test_received_operation_reply_survives_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    binding = client.context.session.bind()
    client.transport.replies[method] = {
        "ok": True,
        "result": {"operation_id": 7, "status": "started"},
    }
    real_send = client.transport.send_line

    def send(payload: dict[str, Any]) -> None:
        real_send(payload)
        client.context.bridge.disconnect()

    monkeypatch.setattr(client.transport, "send_line", send)
    assert client.call("rpc_call", {"method": method, "params": {"tab_id": "t"}}) == {
        "handle": 1,
        "status": "started",
    }
    with pytest.raises(GuiRpcError) as error:
        binding.read_internal("operation.active", {})
    assert error.value.reason == "connection_lost"
    assert not second.sent
    client.context.session.connect_to_gui(port=9912, launch="never", clean=False)
    assert client.context.session.bind().expose_operation(7) == 2


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
        pending = pool.submit(
            binding.send_gui_rpc, "context.labels", {}, timeout_seconds=1.0
        )
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
    prime_operation_discovery(client)
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


def test_rpc_admission_rechecks_queued_intent_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = make_serialization_client(tmp_path, monkeypatch)
    entered, release, attempted = Event(), Event(), Event()
    preceding_replied, revoked = Event(), Event()
    admissions: list[str] = []

    def reply(params: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        assert release.wait(3), "test did not release the preceding RPC"
        preceding_replied.set()
        return {"ok": True, "result": {"operation_id": 1}}

    client.transport.replies["test.start"] = reply
    binding = client.context.session.bind()

    def admit() -> None:
        assert preceding_replied.is_set(), "admission ran before acquiring the RPC lock"
        admissions.append("checked")
        if revoked.is_set():
            raise ValueError("execution cancelled")

    def queued_call() -> dict[str, Any]:
        attempted.set()
        return binding.send_gui_rpc("test.start", {}, before_send=admit)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(binding.send_gui_rpc, "test.start", {})
        try:
            assert entered.wait(3)
            queued = pool.submit(queued_call)
            assert attempted.wait(3)
            revoked.set()
        finally:
            release.set()
        assert first.result(timeout=3)["handle"] == 1
        with pytest.raises(ValueError, match="execution cancelled"):
            queued.result(timeout=3)
    assert admissions == ["checked"]
    assert [method for method, _ in client.transport.sent].count("test.start") == 1

    revoked.clear()
    assert binding.send_gui_rpc("test.start", {}, before_send=admit)["handle"] == 1
    assert admissions == ["checked", "checked"]


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
