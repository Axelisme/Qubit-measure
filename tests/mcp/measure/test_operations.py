"""Public operation tools use the live GUI, including GUI-origin handles."""

from pathlib import Path
from typing import Any

import pytest

from ._support import MeasureClient, make_client


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


def test_status_indexes_gui_origin_operations_without_an_agent_start(
    tmp_path: Path,
) -> None:
    replies: dict[str, dict[str, Any]] = {
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

    assert client.call("status", {}) == {
        "project": {"chip": "chip", "qubit": "qubit", "resonator": "res"},
        "soc": {"connected": True, "mock": True},
        "context": {"active": "bias"},
        "devices": [{"name": "flux", "connected": True}],
        "predictor": {"loaded": False},
        "ready": {"can_run": True, "missing": []},
        "tabs": [{"tab": "gui-tab", "experiment": "ramsey", "running": False}],
        "running": [
            {"op": 1, "tab": "gui-tab", "kind": "analyze"},
            {"op": 2, "tab": None, "kind": "device"},
        ],
    }
    assert ("operation.active", {}) in client.transport.sent


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


def test_wait_rejects_bad_timeout_without_sending_an_operation(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for timeout in (-1, 301, float("nan"), True):
        with pytest.raises(ValueError, match="timeout"):
            client.call("wait", {"op": 1, "timeout": timeout})
    assert not any(method == "operation.await" for method, _ in client.transport.sent)


def test_cancel_short_wait_observes_stop_and_respects_non_cancellable(
    tmp_path: Path,
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
            "status": "cancelled",
            "feedback": "Stop requested",
        },
    }
    assert client.call("cancel", {"op": run_op}) == {"status": "cancelled"}
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
