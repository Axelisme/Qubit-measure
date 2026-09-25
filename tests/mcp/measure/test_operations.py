"""Public operation tools use the live GUI, including GUI-origin handles."""

from pathlib import Path
from typing import Any

import pytest

from ._support import make_client


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
            {"op": 31, "tab": "gui-tab", "kind": "analyze"},
            {"op": 32, "tab": None, "kind": "device"},
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
    assert client.call("connect", {"port": 9912})["status"]["running"] == [
        {"op": 31, "tab": "gui-tab", "kind": "run"}
    ]
    client.context.bridge.disconnect()
    assert client.call("connect", {"port": 9912})["status"]["running"] == [
        {"op": 42, "tab": "gui-tab", "kind": "run"}
    ]
    assert client.call("status", {})["running"][0]["op"] == 42
    with pytest.raises(RuntimeError) as exc_info:
        client.call("wait", {"op": 31})
    assert getattr(exc_info.value, "reason", None) == "unknown_op"
    assert client.call("wait", {"op": 42})["status"] == "finished"
    assert not any(method == "operation.await" for method, _ in first.sent)
    assert [
        params["operation_id"]
        for method, params in second.sent
        if method == "operation.await"
    ] == [31, 42]


def test_wait_timeout_and_feedback_are_running_results(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    for wire, feedback in (
        ({"reason": "timeout"}, None),
        ({"reason": "user_feedback", "feedback": "check frequency"}, "check frequency"),
    ):
        client.transport.replies["operation.await"] = {"ok": True, "result": wire}
        client.transport.replies["operation.progress"] = {
            "ok": True,
            "result": {"active": True, "bars": [{"percent": 35.0, "eta_s": 12.5}]},
        }
        result = client.call("wait", {"op": 31, "timeout": 0})
        assert result["status"] == "running"
        assert result["elapsed_s"] >= 0
        assert result["progress"] == [{"percent": 35.0, "eta_s": 12.5}]
        assert result["eta_s"] == 12.5
        assert result.get("feedback") == feedback


def test_wait_reports_failed_outcome_as_data_and_unknown_as_error(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    client.transport.replies["operation.await"] = {
        "ok": True,
        "result": {
            "reason": "completed",
            "status": "failed",
            "error": {"reason": "failed", "message": "ramp failed"},
        },
    }
    result = client.call("wait", {"op": 32})
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
    assert client.call("cancel", {"op": 31}) == {"status": "cancelled"}
    client.transport.replies["operation.cancel"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "reason": "not_cancellable",
            "message": "post/save cannot cancel",
        },
    }
    with pytest.raises(RuntimeError) as exc_info:
        client.call("cancel", {"op": 32})
    assert getattr(exc_info.value, "reason", None) == "not_cancellable"
