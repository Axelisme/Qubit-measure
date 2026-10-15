"""Connection-bound calls cannot cross GUI incarnations or overlap RPC state."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

import pytest
from zcu_tools.mcp.measure import tools_operation
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.session import GuiConnection, GuiRpcError

from ._save_report_support import analysis_report
from ._support import MeasureClient, WireTransport, full_execution_reply, make_client


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


def _assert_admitted_analysis_connection_loss(reply) -> None:
    assert reply.is_error is True
    assert reply.data["op"] == 1
    assert reply.data["status"] in {"interactive", "failed"}
    error = reply.data["error"] or reply.data["interaction"]["delivery_error"]
    assert "connection" in str(error).lower()


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
            "apply_writeback",
            {"tab": "t"},
            {"tab.get_analyze_result": {"summary": {"value": 1}}},
            "tab.get_analyze_result",
            id="accept-before-writes",
        ),
        pytest.param(
            "apply_writeback",
            {"tab": "t"},
            {
                "tab.get_analyze_result": {"summary": {"value": 1}},
                "tab.get_post_analyze_result": {"summary": {"value": 2}},
                "tab.writeback_preview": {
                    "has_draft": True,
                    "items": [{"id": "x", "target_name": "frequency"}],
                    "destination_context": {},
                },
                "tab.writeback_write": {
                    "written": [
                        {
                            "id": "x",
                            "kind": "parameter",
                            "target": "frequency",
                            "before": 1,
                            "after": 2,
                        }
                    ]
                },
            },
            "tab.writeback_write",
            id="accept-confirmed-primary",
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
    if tool == "apply_writeback":
        client.transport.replies["tab.writeback_preview"] = lambda params: {
            "ok": True,
            "result": {
                "has_draft": True,
                "items": [
                    {"id": "x", "target_name": f"{params['subtab_id']}.frequency"}
                ],
                "destination_context": {},
            },
        }
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
    if tool == "apply_writeback":
        result = client.call(tool, arguments)
        assert result["status"] == "failed"
        assert result["error"]["reason"] == "connection_lost"
        assert result["failed_stage"] == "post"
        assert result["failed_stage_may_have_partial_writes"] is False
        if cut_after == "tab.writeback_write":
            assert result["completed"] == [
                {
                    "stage": "primary",
                    "written": [
                        {
                            "id": "x",
                            "kind": "parameter",
                            "target": "frequency",
                            "before": 1,
                            "after": 2,
                        }
                    ],
                }
            ]
            assert result["not_started"] == []
        else:
            assert result["completed"] == []
            assert result["not_started"] == ["primary"]
    else:
        _assert_admitted_analysis_connection_loss(client.call(tool, arguments))
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


@pytest.mark.parametrize("change", ["eof", "reconnect"])
@pytest.mark.parametrize("cut_after", ["tab.get_analyze_result", "tab.save_artifacts"])
def test_background_analysis_retains_received_facts_without_cross_generation_replay(
    tmp_path, monkeypatch, change, cut_after
):
    client, second = make_restartable_client(tmp_path, monkeypatch)
    session = client.context.session
    replies = {
        "tab.analyze": {
            "operation_id": 71,
            "interactive": False,
            "params": {},
            "invalidated_on_success": [],
        },
        "operation.await": {"reason": "completed", "status": "finished"},
        "tab.get_analyze_result": {
            "summary": {"peak": 5},
            "invalid": [],
            "params": {},
            "operation_state": {
                "analysis_state": {"figure_names": ["fit", "residual"]}
            },
        },
        "tab.save_artifacts": asdict(analysis_report("analysis", ["fit", "residual"])),
    }
    client.transport.replies.update(
        {key: {"ok": True, "result": value} for key, value in replies.items()}
    )
    second.replies["tab.analyze"] = {"ok": True, "result": replies["tab.analyze"]}
    real_send = GuiConnection.send_gui_rpc

    def send(connection, method, params, *args, **kwargs):
        reply = real_send(connection, method, params, *args, **kwargs)
        if method == cut_after:
            if change == "eof":
                client.context.bridge.disconnect()
            else:
                session.connect_to_gui(port=9912, launch="never", clean=False)
        return reply

    monkeypatch.setattr(GuiConnection, "send_gui_rpc", send)
    try:
        reply = full_execution_reply(client, client.call("tab_analyze", {"tab": "t"}))
        assert reply.is_error is True
        result = reply.data
        assert result["error"]["reason"] == "connection_lost"
        assert result["op"] == 1
        assert result["result"]["summary"] == {"peak": 5}
        assert result["save_report"] == (
            asdict(analysis_report("analysis", ["fit", "residual"]))
            if cut_after == "tab.save_artifacts"
            else None
        )
        assert (
            client.call("status", {"execution": result["execution"], "detail": "full"})
            == result
        )
        assert (
            client.call("cancel", {"execution": result["execution"]}).data[
                "gui_cancel"
            ]["status"]
            == "not_needed"
        )
        if change == "eof":
            session.connect_to_gui(port=9912, launch="never", clean=False)
        new_op = client.call(
            "rpc_call",
            {"method": "tab.analyze", "params": {"tab_id": "t", "updates": {}}},
        )
        assert new_op["handle"] == 2
        assert client.call("status", {"execution": result["execution"]})["op"] == 1
        assert not any(
            method
            in {
                "tab.save_artifacts",
                "tab.get_analyze_result",
                "tab.get_figure",
                "operation.cancel",
            }
            for method, _ in second.sent
        )
    finally:
        session.close()


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
        # EOF, not a transport deadline, ends the dispatched RPC; 30s guards deadlock.
        pending = pool.submit(
            binding.send_gui_rpc, "context.labels", {}, timeout_seconds=30.0
        )
        try:
            # The submitted RPC records transport dispatch before EOF or close. The 30s limit only guards deadlock.
            assert sent.wait(30)
            client.transport.close()
            assert client.transport.on_closed is not None
            client.transport.on_closed(None)
            with pytest.raises(GuiRpcError) as error:
                # The submitted call completes after release/close; 30s only guards deadlock.
                pending.result(timeout=30)
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
        ("rpc_call", {"method": "adapter.list"}),
        ("rpc_describe", {"method": "adapter.list"}),
        ("rpc_list", {}),
    ],
)
def test_assembled_tools_keep_an_inherited_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    arguments: dict[str, Any],
) -> None:
    client, second = make_restartable_client(tmp_path, monkeypatch)
    tools = build_measure_tools(
        client.context.bound(), recipes=client.context.session.recipes.definitions
    )
    client.context.session.connect_to_gui(port=9912, launch="never", clean=False)
    sent = list(second.sent)
    with pytest.raises(GuiRpcError) as error:
        tools[tool]["handler"](arguments)
    assert error.value.reason == "connection_lost"
    assert second.sent == sent


def test_operation_discovery_does_not_expose_an_old_reply_in_a_new_gui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        client.call("status", {})
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


def record_serial_transport(
    monkeypatch: pytest.MonkeyPatch, *transports: WireTransport
) -> tuple[Lock, list[tuple[str, str]]]:
    """Record transport entry/exit and reject overlapping sends across incarnations."""
    active = Lock()
    trace: list[tuple[str, str]] = []

    def observe(transport: WireTransport) -> None:
        send_line = transport.send_line

        def send(payload: dict[str, Any]) -> None:
            assert active.acquire(blocking=False), "transport calls overlapped"
            method = payload["method"]
            trace.append(("enter", method))
            try:
                send_line(payload)
            finally:
                trace.append(("exit", method))
                active.release()

        monkeypatch.setattr(transport, "send_line", send)

    for transport in transports:
        observe(transport)
    return active, trace


def test_rpc_admission_rechecks_queued_intent_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = make_serialization_client(tmp_path, monkeypatch)
    entered, release, attempted = Event(), Event(), Event()
    preceding_replied, revoked, finished = Event(), Event(), Event()
    admissions: list[str] = []

    def reply(params: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        # The test finally releases this responder barrier. The 30s limit only guards deadlock.
        assert release.wait(30), "test did not release the preceding RPC"
        preceding_replied.set()
        return {"ok": True, "result": {"operation_id": 1}}

    client.transport.replies["test.start"] = reply
    binding = client.context.session.bind()

    _, trace = record_serial_transport(monkeypatch, client.transport)

    def admit() -> None:
        assert preceding_replied.is_set(), "admission ran before acquiring the RPC lock"
        admissions.append("checked")
        if revoked.is_set():
            raise ValueError("execution cancelled")

    def queued_call() -> dict[str, Any]:
        attempted.set()
        try:
            return binding.send_gui_rpc("test.start", {}, before_send=admit)
        finally:
            finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(binding.send_gui_rpc, "test.start", {})
        try:
            # The submitted RPC reaches the recording responder. The 30s limit only guards deadlock.
            assert entered.wait(30)
            queued = pool.submit(queued_call)
            # The submitted competitor records its attempt before calling the session. The 30s limit only guards deadlock.
            assert attempted.wait(30)
            # No public queue-arrival point: this window affects detection only, never load correctness.
            assert not finished.wait(0.1), "queued admission crossed the active RPC"
            revoked.set()
        finally:
            release.set()
        # The submitted call completes after release/close; 30s only guards deadlock.
        assert first.result(timeout=30)["handle"] == 1
        with pytest.raises(ValueError, match="execution cancelled"):
            # The submitted call completes after release/close; 30s only guards deadlock.
            queued.result(timeout=30)
    assert finished.is_set()
    assert trace == [("enter", "test.start"), ("exit", "test.start")]
    assert admissions == ["checked"]
    assert [method for method, _ in client.transport.sent].count("test.start") == 1

    revoked.clear()
    assert binding.send_gui_rpc("test.start", {}, before_send=admit)["handle"] == 1
    assert admissions == ["checked", "checked"]


def record_serial_connection(
    monkeypatch: pytest.MonkeyPatch, client: MeasureClient, second: WireTransport
) -> tuple[list[float], list[tuple[str, str]]]:
    """Observe RPC timeouts and connection replacement without changing either."""
    observed_timeouts: list[float] = []
    real_send = client.context.bridge.send_rpc_raw

    def send(
        method: str, params: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        if method == "test.start":
            observed_timeouts.append(timeout_seconds)
        return real_send(method, params, timeout_seconds)

    monkeypatch.setattr(client.context.bridge, "send_rpc_raw", send)
    active, trace = record_serial_transport(monkeypatch, client.transport, second)
    real_connect = client.context.bridge.connect

    def connect(port: int, token: str | None = None) -> str:
        assert not active.locked(), "connect crossed the active transport call"
        trace.append(("connect", str(port)))
        return real_connect(port, token)

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    return observed_timeouts, trace


@pytest.mark.parametrize("action", ["rpc", "connect"])
def test_rpc_and_connect_serialize_catalog_timeout_and_handle_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    client, second = make_serialization_client(tmp_path, monkeypatch)
    entered, release, attempted, finished = Event(), Event(), Event(), Event()
    calls = 0

    def reply(params: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            # The test finally releases this responder barrier. The 30s limit only guards deadlock.
            assert release.wait(30), "test did not release the first RPC"
        return {"ok": True, "result": {"operation_id": calls}}

    client.transport.replies["test.start"] = reply
    binding = client.context.session.bind()
    observed_timeouts, trace = record_serial_connection(monkeypatch, client, second)

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
            # The submitted RPC reaches the recording responder. The 30s limit only guards deadlock.
            assert entered.wait(30)
            other = pool.submit(competing_call)
            # The submitted competitor records its attempt before calling the session. The 30s limit only guards deadlock.
            assert attempted.wait(30)
            # No public queue-arrival point: this window affects detection only, never load correctness.
            assert not finished.wait(0.1), "competing call crossed the active RPC"
        finally:
            release.set()
        # The submitted call completes after release/close; 30s only guards deadlock.
        old_result = first.result(timeout=30)
        # The submitted call completes after release/close; 30s only guards deadlock.
        other_result = other.result(timeout=30)
    assert finished.is_set()
    assert trace[:2] == [("enter", "test.start"), ("exit", "test.start")]
    assert old_result["handle"] == 1
    if action == "rpc":
        assert trace[2:] == [("enter", "test.start"), ("exit", "test.start")]
        assert other_result["handle"] == 2
        assert observed_timeouts == [6.0, 6.0]
        assert client.context.session.operation_handles == {"job:old": 2}
    else:
        assert trace[2] == ("connect", "9912")
        assert ("enter", "rpc.catalog") in trace[3:]
        assert ("exit", "rpc.catalog") in trace[3:]
        new = client.context.session.bind().send_gui_rpc("test.start", {})
        assert new["handle"] == 2
        assert observed_timeouts == [6.0, 10.0]
        assert client.context.session.operation_handles == {"job:new": 2}
        with pytest.raises(GuiRpcError) as error:
            binding.send_gui_rpc("test.start", {})
        assert error.value.reason == "connection_lost"
