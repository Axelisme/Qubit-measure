"""Public operation tools use the live GUI, including GUI-origin handles."""

from pathlib import Path
from typing import Any

import pytest

from ._support import MeasureClient, RpcResponder, WireTransport, make_client


class DisconnectAfterProbe(WireTransport):
    """Close the old GUI immediately after a connection-status probe."""

    def __init__(self, responder: RpcResponder) -> None:
        self.disconnect_after_probe = False
        super().__init__(responder)

    @property
    def is_open(self) -> bool:
        was_open = self._open
        if self.disconnect_after_probe:
            self.disconnect_after_probe = False
            self._open = False
        return was_open

    @is_open.setter
    def is_open(self, value: bool) -> None:
        self._open = value


def _restartable_operation_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[MeasureClient, DisconnectAfterProbe, WireTransport, int]:
    def gui_reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        replies = {
            "state.has_project": {"value": False},
            "state.has_active_context": {"value": False},
            "state.has_soc": {"value": False},
            "context.active": {"label": None},
            "device.list": {"devices": []},
            "predictor.info": {"loaded": False},
            "tab.snapshot": {"tabs": []},
            "operation.active": {
                "operations": [{"op": 1, "tab": "gui-tab", "kind": "run"}]
            },
        }
        return replies[method]

    client = make_client(tmp_path, gui_reply, port_is_open=lambda port: True)
    first = DisconnectAfterProbe(gui_reply)
    second = WireTransport(gui_reply)
    for transport in (first, second):
        transport.replies["rpc.catalog"] = client.transport.replies["rpc.catalog"]
    client.transport = first
    client.context.bridge.set_transport(None)
    transports = iter((first, second))

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    old_handle = client.call("connect", {"port": 9912})["status"]["running"][0]["op"]
    return client, first, second, old_handle


def discover_operation(client: MeasureClient, gui_id: int) -> int:
    """Discover a GUI-owned op through the same status path as an agent."""
    client.transport.replies.update(
        {
            "state.has_project": {"ok": True, "result": {"value": False}},
            "state.has_active_context": {"ok": True, "result": {"value": False}},
            "state.has_soc": {"ok": True, "result": {"value": False}},
            "context.active": {"ok": True, "result": {"label": None}},
            "device.list": {"ok": True, "result": {"devices": []}},
            "predictor.info": {"ok": True, "result": {"loaded": False}},
            "tab.snapshot": {"ok": True, "result": {"tabs": []}},
            "operation.active": {
                "ok": True,
                "result": {
                    "operations": [{"op": gui_id, "tab": None, "kind": "device"}]
                },
            },
        }
    )
    return client.call("status", {})["running"][0]["op"]


@pytest.mark.parametrize("analysis_count", [0, 2])
def test_status_indexes_gui_operations_and_session_executions(
    tmp_path: Path, analysis_count: int
) -> None:
    replies: dict[str, dict[str, Any]] = {
        "tab.writeback_preview": {
            "has_draft": False,
            "items": [],
            "destination_context": {},
        },
        "state.has_project": {"value": True},
        "project.info": {"chip_name": "chip", "qub_name": "qubit", "res_name": "res"},
        "state.has_context": {"value": True},
        "state.has_active_context": {"value": True},
        "context.active": {"label": "bias"},
        "state.has_soc": {"value": True},
        "soc.info": {"is_mock": True},
        "device.list": {"devices": [{"name": "flux", "status": "connected"}]},
        "predictor.info": {"loaded": False},
        "tab.snapshot": {
            "tabs": [
                {
                    "tab_id": "gui-tab",
                    "adapter_name": "ramsey",
                    "interaction": {"is_running": False},
                }
            ]
        },
        "operation.active": {
            "operations": [
                {"op": 31, "tab": "gui-tab", "kind": "analyze"},
                {"op": 32, "tab": None, "kind": "device"},
            ]
        },
    }
    client = make_client(tmp_path, lambda method, params: replies[method])
    executions = []
    for index in range(analysis_count):
        replies["tab.analyze"] = {
            "operation_id": 71 + index,
            "interactive": False,
            "params": {},
            "invalidated_on_success": [],
        }
        replies["operation.await"] = {"reason": "completed", "status": "finished"}
        replies["tab.get_analyze_result"] = {
            "summary": None,
            "invalid": [],
            "params": {},
            "operation_state": {"analysis_state": {"figure_names": []}},
        }
        executions.append(client.call("tab_analyze", {"tab": "gui-tab"}).data)

    pending = client.context.session.executions.start(
        client.context.session.bind(), "pending-tab", "post"
    )
    pending_summary = client.call("status", {"execution": pending.snapshot().execution})
    assert client.call("status", {}) == {
        "project": {"chip": "chip", "qubit": "qubit", "resonator": "res"},
        "soc": {"connected": True, "mock": True},
        "context": {"active": "bias"},
        "devices": [{"name": "flux", "status": "connected", "connected": True}],
        "predictor": {"loaded": False},
        "ready": {"can_run": True, "missing": []},
        "tabs": [{"tab": "gui-tab", "experiment": "ramsey", "running": False}],
        "executions": [pending_summary],
        "terminal_count": analysis_count,
        "query_hint": 'Use status(execution=<id>, detail="full") for completed executions.',
        "running": [
            {"op": analysis_count + 1, "tab": "gui-tab", "kind": "analyze"},
            {"op": analysis_count + 2, "tab": None, "kind": "device"},
        ],
    }
    assert ("operation.active", {}) in client.transport.sent
    changed = client.call("status", {})
    changed["executions"].clear()
    assert client.call("status", {})["executions"] == [pending_summary]
    before = len(client.transport.sent)
    for item in executions:
        full = client.call("status", {"execution": item["execution"], "detail": "full"})
        assert full["status"] == "finished"
        assert full["execution"] == item["execution"]
    assert len(client.transport.sent) == before
    client.context.session.close()


def test_reconnect_indexes_new_gui_operations_without_reusing_old_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gui_reply(op: int):
        replies = {
            "state.has_project": {"value": False},
            "state.has_active_context": {"value": False},
            "state.has_soc": {"value": False},
            "context.active": {"label": None},
            "device.list": {"devices": []},
            "predictor.info": {"loaded": False},
            "tab.snapshot": {
                "tabs": [
                    {
                        "tab_id": "gui-tab",
                        "adapter_name": "fake",
                        "interaction": {"is_running": True},
                    }
                ]
            },
            "operation.active": {
                "operations": [{"op": op, "tab": "gui-tab", "kind": "run"}]
            },
        }
        return lambda method, params: replies[method]

    client = make_client(tmp_path, gui_reply(31), port_is_open=lambda port: True)
    first = client.transport
    second = type(first)(gui_reply(42))
    second.replies["rpc.catalog"] = first.replies["rpc.catalog"]
    second.replies["operation.await"] = lambda params: (
        {
            "ok": False,
            "error": {
                "code": "invalid_params",
                "reason": "unknown_op",
                "message": "unknown operation",
            },
        }
        if params["operation_id"] == 31
        else {"ok": True, "result": {"reason": "completed", "status": "finished"}}
    )
    client.context.bridge.set_transport(None)
    transports = iter((first, second))

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    old = client.call("connect", {"port": 9912})["status"]["running"][0]["op"]
    client.context.bridge.disconnect()
    new = client.call("connect", {"port": 9912})["status"]["running"][0]["op"]
    assert new != old
    assert client.call("status", {})["running"][0]["op"] == new
    with pytest.raises(RuntimeError) as exc_info:
        client.call("wait", {"op": old})
    assert getattr(exc_info.value, "reason", None) == "unknown_op"
    assert client.call("wait", {"op": new})["status"] == "finished"
    assert not any(method == "operation.await" for method, _ in first.sent)
    assert [
        params["operation_id"]
        for method, params in second.sent
        if method == "operation.await"
    ] == [42]


def test_reconnect_never_reuses_an_exposed_handle_for_the_same_gui_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gui_reply(method: str, params: dict[str, Any]) -> dict[str, Any]:
        replies = {
            "state.has_project": {"value": False},
            "state.has_active_context": {"value": False},
            "state.has_soc": {"value": False},
            "context.active": {"label": None},
            "device.list": {"devices": []},
            "predictor.info": {"loaded": False},
            "tab.snapshot": {"tabs": []},
            "operation.active": {
                "operations": [{"op": 1, "tab": "gui-tab", "kind": "run"}]
            },
        }
        return replies[method]

    client = make_client(tmp_path, gui_reply, port_is_open=lambda port: True)
    first = client.transport
    second = type(first)(gui_reply)
    second.replies["rpc.catalog"] = first.replies["rpc.catalog"]
    second.replies["operation.await"] = {
        "ok": True,
        "result": {"reason": "completed", "status": "finished"},
    }
    second.replies["operation.cancel"] = {
        "ok": True,
        "result": {"status": "finished"},
    }
    client.context.bridge.set_transport(None)
    transports = iter((first, second))

    def connect(port: int, token: str | None = None) -> str:
        client.context.bridge.set_transport(next(transports))
        return "connected"

    monkeypatch.setattr(client.context.bridge, "connect", connect)
    old_op = client.call("connect", {"port": 9912})["status"]["running"][0]["op"]
    client.context.bridge.disconnect()
    # The first call after a restart must invalidate the old handle before RPC.
    with pytest.raises(RuntimeError) as exc_info:
        client.call("wait", {"op": old_op})
    assert getattr(exc_info.value, "reason", None) == "unknown_op"
    new_op = client.call("connect", {"port": 9912})["status"]["running"][0]["op"]
    assert new_op != old_op
    for tool in ("wait", "cancel"):
        with pytest.raises(RuntimeError) as exc_info:
            client.call(tool, {"op": old_op})
        assert getattr(exc_info.value, "reason", None) == "unknown_op"
    assert client.call("wait", {"op": new_op})["status"] == "finished"
    assert client.call("cancel", {"op": new_op})["status"] == "finished"
    assert [
        params["operation_id"]
        for method, params in second.sent
        if method in ("operation.await", "operation.cancel")
    ] == [1, 1]


@pytest.mark.parametrize("tool", ("wait", "cancel"))
def test_old_operation_handle_never_targets_reused_id_after_lookup_disconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    client, first, second, old_handle = _restartable_operation_client(
        tmp_path, monkeypatch
    )
    second.replies["operation.await"] = {
        "ok": True,
        "result": {"reason": "completed", "status": "finished"},
    }
    second.replies["operation.cancel"] = {"ok": True, "result": {"status": "finished"}}

    first.disconnect_after_probe = True
    with pytest.raises(RuntimeError, match="unknown|expired|not connected"):
        client.call(tool, {"op": old_handle})
    assert not any(
        method in ("operation.await", "operation.cancel") for method, _ in second.sent
    ), second.sent
    assert client.call("status", {})["running"][0]["op"] != old_handle


def test_wait_does_not_read_reused_id_progress_after_gui_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, first, second, old_handle = _restartable_operation_client(
        tmp_path, monkeypatch
    )

    def timeout_then_disconnect(params: dict[str, Any]) -> dict[str, Any]:
        first.close()
        return {"ok": True, "result": {"reason": "timeout"}}

    first.replies["operation.await"] = timeout_then_disconnect
    second.replies["operation.progress"] = {
        "ok": True,
        "result": {"active": True, "bars": [{"percent": 90.0}]},
    }

    with pytest.raises(RuntimeError, match="unknown|expired|not connected"):
        client.call("wait", {"op": old_handle, "timeout": 0})
    assert not any(method == "operation.progress" for method, _ in second.sent), (
        second.sent
    )
    assert any(method == "operation.await" for method, _ in first.sent)
    assert client.call("status", {})["running"][0]["op"] != old_handle


def test_started_operation_and_gui_status_share_one_agent_handle(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    client.transport.replies["device.connect"] = {
        "ok": True,
        "result": {"operation_id": 31},
    }
    reply = client.call(
        "rpc_call",
        {
            "method": "device.connect",
            "params": {"type_name": "FakeDevice", "name": "flux", "address": "fake"},
        },
    )
    from_start = reply["handle"]
    from_status = discover_operation(client, 31)
    assert from_start == from_status
    client.transport.replies["operation.await"] = {
        "ok": True,
        "result": {"reason": "completed", "status": "finished"},
    }
    assert client.call("wait", {"op": from_status})["status"] == "finished"
    assert (
        "operation.await",
        {"operation_id": 31, "timeout": 60},
    ) in client.transport.sent


def test_wait_timeout_and_feedback_are_running_results(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    op = discover_operation(client, 31)
    for wire, feedback in (
        ({"reason": "timeout"}, None),
        ({"reason": "user_feedback", "feedback": "check frequency"}, "check frequency"),
    ):
        client.transport.replies["operation.await"] = {"ok": True, "result": wire}
        client.transport.replies["operation.progress"] = {
            "ok": True,
            "result": {"active": True, "bars": [{"percent": 35.0, "eta_s": 12.5}]},
        }
        result = client.call("wait", {"op": op, "timeout": 0})
        assert result["status"] == "running"
        assert result["elapsed_s"] >= 0
        assert result["progress"] == [{"percent": 35.0, "eta_s": 12.5}]
        assert result["eta_s"] == 12.5
        assert result.get("feedback") == feedback


@pytest.mark.parametrize(
    "reason", [None, "Stop requested", "Autofluxdep run stop requested"]
)
def test_wait_delivers_cancelled_stop_reason(
    tmp_path: Path, reason: str | None
) -> None:
    client = make_client(tmp_path)
    op = discover_operation(client, 31)
    wire = {"reason": "completed", "status": "cancelled"}
    if reason is not None:
        wire["feedback"] = reason
    client.transport.replies["operation.await"] = {"ok": True, "result": wire}

    result = client.call("wait", {"op": op, "timeout": 0})
    elapsed = result.pop("elapsed_s")
    assert elapsed >= 0
    expected = {"status": "cancelled"}
    if reason is not None:
        expected["feedback"] = reason
    assert result == expected


def test_wait_reports_failed_outcome_as_data_and_unknown_as_error(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    op = discover_operation(client, 32)
    client.transport.replies["operation.await"] = {
        "ok": True,
        "result": {
            "reason": "completed",
            "status": "failed",
            "error": {"reason": "failed", "message": "ramp failed"},
        },
    }
    result = client.call("wait", {"op": op})
    assert result["status"] == "failed"
    assert result["error"] == {"reason": "failed", "message": "ramp failed"}
    client.transport.replies["operation.await"] = {
        "ok": False,
        "error": {
            "code": "invalid_params",
            "reason": "unknown_op",
            "message": "unknown or evicted op",
        },
    }
    with pytest.raises(RuntimeError) as exc_info:
        client.call("wait", {"op": 999})
    assert getattr(exc_info.value, "reason", None) == "unknown_op"
    assert ("operation.progress", {"operation_id": 999}) not in client.transport.sent


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        *[
            (tool, arguments, "exactly one")
            for tool in ("wait", "cancel", "finish_early")
            for arguments in (
                {},
                {"op": 1, "execution": "analysis-1"},
                {"op": 1, "execution": None},
            )
        ],
        *[
            (tool, {"execution": value}, "non-empty string")
            for tool in ("status", "wait", "cancel", "finish_early")
            for value in ("", None, True, 1)
        ],
        *[
            ("status", {"execution": "analysis-1", "detail": detail}, "detail")
            for detail in ("unknown", None, True, 1, {})
        ],
        ("status", {"detail": "full"}, "requires execution"),
        *[
            ("wait", {"execution": "analysis-1", "timeout": timeout}, "timeout")
            for timeout in (-1, 301, True, float("nan"), float("inf"), "1")
        ],
    ],
)
def test_execution_query_rejects_invalid_input_without_gui_access(
    tmp_path: Path, tool: str, arguments: dict[str, Any], message: str
) -> None:
    client = make_client(tmp_path)
    try:
        with pytest.raises(ValueError, match=message):
            client.call(tool, arguments)
        assert client.transport.sent == []
    finally:
        client.context.session.close()


@pytest.mark.parametrize("tool", ["status", "wait", "cancel", "finish_early"])
@pytest.mark.parametrize("execution", ["analysis-missing", "recipe-missing"])
def test_unknown_execution_is_a_query_failure_without_gui_access(
    tmp_path: Path, tool: str, execution: str
) -> None:
    client = make_client(tmp_path)
    try:
        with pytest.raises(RuntimeError) as error:
            client.call(tool, {"execution": execution})
        assert getattr(error.value, "reason", None) == "unknown_execution"
        assert client.transport.sent == []
    finally:
        client.context.session.close()


def test_wait_rejects_bad_timeout_without_sending_an_operation(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for timeout in (-1, 301, float("nan"), True):
        with pytest.raises(ValueError, match="timeout"):
            client.call("wait", {"op": 1, "timeout": timeout})
    assert not any(method == "operation.await" for method, _ in client.transport.sent)


@pytest.mark.parametrize("outcome", ["cancelled", "finished", "failed"])
def test_cancel_reports_the_gui_request_and_wait_observes_the_outcome(
    tmp_path: Path, outcome: str
) -> None:
    client = make_client(tmp_path)
    run_op = discover_operation(client, 31)
    other_op = discover_operation(client, 32)
    client.transport.replies["operation.cancel"] = {
        "ok": True,
        "result": {"status": "cancelling"},
    }
    client.transport.replies["operation.await"] = {
        "ok": True,
        "result": {
            "reason": "completed",
            "status": outcome,
            "feedback": "Stop requested",
            "error": {"reason": "failed", "message": "ramp failed"},
        },
    }
    receipt = client.call("cancel", {"op": run_op})
    assert receipt == {
        "execution": None,
        "op": run_op,
        "status": "cancelling",
        "cancel_requested": True,
    }
    terminal = client.call("wait", {"op": run_op, "timeout": 2})
    assert terminal["status"] == outcome
    if outcome == "failed":
        assert terminal["error"]["message"] == "ramp failed"
    client.transport.replies["operation.cancel"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "not_cancellable",
            "message": "post/save cannot cancel",
        },
    }
    with pytest.raises(RuntimeError) as exc_info:
        client.call("cancel", {"op": other_op})
    assert getattr(exc_info.value, "reason", None) == "not_cancellable"
