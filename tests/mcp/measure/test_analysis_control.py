"""Analysis cancellation, Save admission, and execution lifecycle contracts."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Event, Thread
from typing import Any

import pytest
from zcu_tools.mcp.measure.session import GuiConnection

from ._analyze_support import (
    AMBIGUOUS_SAVE_ERRORS as _AMBIGUOUS_SAVE_ERRORS,
)
from ._analyze_support import (
    PNG as _PNG,
)
from ._analyze_support import (
    analysis_client as _client,
)
from ._analyze_support import (
    analysis_result_reply as _result_reply,
)
from ._analyze_support import (
    analysis_start_reply as _start_reply,
)
from ._analyze_support import (
    assert_ambiguous_save_error as _assert_ambiguous_save_error,
)
from ._analyze_support import (
    call_full_execution_stdio as _call_full_execution_stdio,
)
from ._analyze_support import (
    call_stdio as _call_stdio,
)
from ._analyze_support import (
    sent_methods as _methods,
)
from ._analyze_support import (
    stdio_data as _data,
)
from ._save_report_support import analysis_report
from ._support import MeasureClient


@pytest.mark.parametrize("selector", ["op", "execution"])
@pytest.mark.parametrize("outcome", ["finished", "cancelled", "failed"])
@pytest.mark.parametrize("gui_cancel", ["cancelling", "not_cancellable", "failed"])
def test_cancel_latches_intent_and_retains_original_terminal(
    tmp_path, clients, monkeypatch, selector, outcome, gui_cancel
):
    terminal = Event()

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [], interactive=True)
        if method == "tab.interact":
            return {"operation_id": 71, "state": {}, "figure": None}
        if method == "operation.await":
            assert params["operation_id"] == 71
            return (
                {"reason": "completed", "status": outcome, "error": "real failure"}
                if terminal.is_set()
                else {"reason": "timeout", "status": "running"}
            )
        if method == "operation.cancel":
            assert params["operation_id"] == 71
            return {"status": gui_cancel}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    started = _data(
        _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    )
    if gui_cancel in ("failed", "not_cancellable"):
        client.transport.replies["operation.cancel"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": gui_cancel,
                "message": "cannot send cancel",
            },
        }
    args = {selector: started[selector]}
    cancelled = _call_stdio(monkeypatch, client, "cancel", args)
    assert bool(cancelled.get("isError")) is (gui_cancel == "failed")
    data = json.loads(cancelled["content"][0]["text"])
    assert set(data) == {
        "execution",
        "op",
        "run_op",
        "status",
        "phase",
        "cancel_requested",
        "finish_early_requested",
        "gui_cancel",
    }
    assert data["op"] == started["op"]
    assert data["run_op"] is None
    assert data["status"] == "running"
    assert data["finish_early_requested"] is False
    assert data["execution"] == started["execution"]
    assert data["cancel_requested"] is True
    assert data["gui_cancel"]["status"] == (
        "requested" if gui_cancel == "cancelling" else gui_cancel
    )
    if gui_cancel == "failed":
        assert "cannot send cancel" in data["gui_cancel"]["error"]["message"]
    repeated = _call_stdio(monkeypatch, client, "cancel", args)
    assert (
        json.loads(repeated["content"][0]["text"])["gui_cancel"] == data["gui_cancel"]
    )
    assert _methods(client).count("operation.cancel") == 1
    terminal.set()
    completed = _data(
        _call_full_execution_stdio(
            monkeypatch,
            client,
            "wait",
            # The released worker completes this execution; 30s only guards deadlock.
            {"execution": started["execution"], "timeout": 30},
        )
    )
    assert completed["status"] == ("failed" if outcome == "failed" else "cancelled")
    assert completed["operation_outcome"]["status"] == outcome
    assert completed["cancel_requested"] is True
    assert completed["result"] is None
    assert completed["save_status"] == "not_started"
    assert set(_methods(client)) <= {
        "tab.analyze",
        "tab.interact",
        "operation.await",
        "operation.cancel",
    }
    frozen = client.call(
        "status", {"execution": started["execution"], "detail": "full"}
    )
    final = _data(_call_stdio(monkeypatch, client, "cancel", args))
    assert final["gui_cancel"]["status"] == "not_needed"
    assert (
        client.call("status", {"execution": started["execution"], "detail": "full"})
        == frozen
    )
    assert _methods(client).count("operation.cancel") == 1


def _admitted_save_terminal(
    client: MeasureClient, outcome: str, names: list[str]
) -> dict[str, Any]:
    if outcome == "lost":
        client.transport.close()
        assert client.transport.on_closed is not None
        client.transport.on_closed(EOFError("save reply lost"))
        return {"ok": True, "result": asdict(analysis_report("analysis", names))}
    if outcome in _AMBIGUOUS_SAVE_ERRORS:
        return {"ok": False, "error": _AMBIGUOUS_SAVE_ERRORS[outcome]}
    if outcome == "rejected":
        return {
            "ok": False,
            "error": {"code": "precondition_failed", "message": "disk rejected export"},
        }
    return {"ok": True, "result": asdict(analysis_report("analysis", names))}


@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize(
    "save_outcome", ["saved", "rejected", "lost", *_AMBIGUOUS_SAVE_ERRORS]
)
def test_cancel_during_admitted_save_retains_the_real_reply(
    tmp_path, clients, monkeypatch, has_residual, save_outcome
):
    saving, release, intent = Event(), Event(), Event()
    names = ["prefix", "fit", "residual"] if has_residual else ["prefix", "fit"]
    read_internal = GuiConnection.read_internal

    def read(connection, method, params, **kwargs):
        if method == "operation.cancel":
            intent.set()
        return read_internal(connection, method, params, **kwargs)

    monkeypatch.setattr(GuiConnection, "read_internal", read)

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", names, {})
        if method == "operation.cancel":
            return {"status": "finished"}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)

    def save(params):
        assert params == {
            "tab_id": "t",
            "artifacts": [{"section": "analysis", "name": "results"}],
            "operation_id": 71,
        }
        saving.set()
        # The test finally releases this responder barrier. The 30s limit only guards deadlock.
        assert release.wait(30), "test did not release admitted save"
        return _admitted_save_terminal(client, save_outcome, names)

    client.transport.replies["tab.save_artifacts"] = save
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            started = _data(
                _call_full_execution_stdio(
                    monkeypatch, client, "tab_analyze", {"tab": "t"}
                )
            )
            # The completion worker reaches Save after the GUI completion is released. The 30s limit only guards deadlock.
            assert saving.wait(30)
            request = pool.submit(
                client.call, "cancel", {"execution": started["execution"]}
            )
            # The submitted cancel call records intent before its GUI RPC. The 30s limit only guards deadlock.
            assert intent.wait(30)
            pending = client.call(
                "status", {"execution": started["execution"], "detail": "full"}
            )
            assert {
                key: pending[key]
                for key in ("cancel_requested", "status", "save_status", "save_report")
            } == {
                "cancel_requested": True,
                "status": "running",
                "save_status": "saving",
                "save_report": None,
            }
            pending_summary = client.call("status", {"execution": started["execution"]})
            assert pending_summary["artifacts"] == {}
        finally:
            release.set()
        # The submitted call completes after release/close; 30s only guards deadlock.
        cancelled = request.result(timeout=30)
    assert cancelled.data["cancel_requested"] is True
    completed = _data(
        _call_full_execution_stdio(
            monkeypatch,
            client,
            "wait",
            # The released worker completes this execution; 30s only guards deadlock.
            {"execution": started["execution"], "timeout": 30},
        )
    )
    assert {
        "status": completed["status"],
        "operation_status": completed["operation_outcome"]["status"],
        "analysis_summary": completed["result"]["summary"],
    } == {
        "status": "cancelled" if save_outcome == "saved" else "failed",
        "operation_status": "finished",
        "analysis_summary": {"frequency": 5.0},
    }
    unknown = save_outcome == "lost" or save_outcome in _AMBIGUOUS_SAVE_ERRORS
    assert completed["save_status"] == (
        "saved" if save_outcome == "saved" else "unknown" if unknown else "incomplete"
    )
    report = (
        asdict(analysis_report("analysis", names)) if save_outcome == "saved" else None
    )
    assert completed["save_report"] == report
    summary = client.call("status", {"execution": started["execution"]})
    assert summary["artifacts"] == (report["artifacts"] if report is not None else {})
    if save_outcome != "saved":
        assert completed["error"]["phase"] == "image_save"
    _assert_ambiguous_save_error(completed, save_outcome)
    assert _methods(client).count("tab.save_artifacts") == 1
    assert "tab.get_figure" not in _methods(client)
    assert "tab.writeback_preview" not in _methods(client)


def test_cancel_rejects_save_queued_behind_another_rpc(tmp_path, clients, monkeypatch):
    save_ready, blocker_entered, release_blocker = Event(), Event(), Event()
    intent, dispatch_save = Event(), Event()
    send_rpc = GuiConnection.send_gui_rpc
    read_internal = GuiConnection.read_internal

    def send(connection, method, params, *args, **kwargs):
        if method == "tab.save_artifacts":
            save_ready.set()
            # The submitted blocker enters the responder before Save can dispatch. The 30s limit only guards deadlock.
            assert blocker_entered.wait(30), "test did not occupy the RPC lock"
            dispatch_save.set()
        return send_rpc(connection, method, params, *args, **kwargs)

    def read(connection, method, params, **kwargs):
        if method == "operation.cancel":
            intent.set()
        return read_internal(connection, method, params, **kwargs)

    monkeypatch.setattr(GuiConnection, "send_gui_rpc", send)
    monkeypatch.setattr(GuiConnection, "read_internal", read)

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {})
        if method == "entry.info":
            blocker_entered.set()
            # The test finally releases the occupied RPC. The 30s limit only guards deadlock.
            assert release_blocker.wait(30), "test did not release the RPC lock"
            return {"project": "test"}
        if method == "operation.cancel":
            return {"status": "finished"}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            started = _data(
                _call_full_execution_stdio(
                    monkeypatch, client, "tab_analyze", {"tab": "t"}
                )
            )
            # The completion worker captured the result and next attempts Save. The 30s limit only guards deadlock.
            assert save_ready.wait(30)
            blocker = pool.submit(
                client.call, "rpc_call", {"method": "entry.info", "params": {}}
            )
            # The submitted blocker releases the pre-Save boundary. The 30s limit only guards deadlock.
            assert dispatch_save.wait(30)
            request = pool.submit(client.call, "cancel", {"op": started["op"]})
            # The submitted cancel call records intent before its GUI RPC. The 30s limit only guards deadlock.
            assert intent.wait(30)
            pending = client.call(
                "status", {"execution": started["execution"], "detail": "full"}
            )
            assert pending["cancel_requested"] is True
            assert pending["save_status"] == "not_started"
            assert pending["save_report"] is None
        finally:
            blocker_entered.set()
            release_blocker.set()
        # The submitted call completes after release/close; 30s only guards deadlock.
        blocker.result(timeout=30)
        # The submitted call completes after release/close; 30s only guards deadlock.
        assert request.result(timeout=30).data["cancel_requested"] is True
    completed = _data(
        _call_full_execution_stdio(
            monkeypatch,
            client,
            "wait",
            # The released worker completes this execution; 30s only guards deadlock.
            {"execution": started["execution"], "timeout": 30},
        )
    )
    assert completed["status"] == "cancelled"
    assert completed["save_report"] is None
    assert completed["save_status"] == "not_started"
    assert "tab.save_artifacts" not in _methods(client)
    assert "tab.get_figure" not in _methods(client)


@pytest.mark.parametrize("interactive", [False, True])
def test_close_after_start_receipt_retains_operation_without_new_work(
    tmp_path, clients, monkeypatch, interactive
):
    send_rpc = GuiConnection.send_gui_rpc

    def respond(method, params):
        assert method == "tab.analyze"
        return _start_reply({}, [], interactive=interactive)

    client = _client(tmp_path, clients, respond)

    def send(connection, method, params, *args, **kwargs):
        reply = send_rpc(connection, method, params, *args, **kwargs)
        if method == "tab.analyze":
            client.context.session.close()
        return reply

    monkeypatch.setattr(GuiConnection, "send_gui_rpc", send)
    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    assert reply["isError"] is True
    stopped = json.loads(reply["content"][0]["text"])
    assert stopped["op"] == 1
    assert stopped["tab"] == "t"
    assert stopped["status"] == "failed"
    assert stopped["error"]["reason"] == "session_closed"
    assert stopped["operation_outcome"] is None
    assert stopped["cancel_requested"] is False
    assert stopped["save_status"] == "not_started"
    assert _methods(client) == ["tab.analyze"]
    assert (
        client.call("status", {"execution": stopped["execution"], "detail": "full"})
        == stopped
    )


@pytest.mark.parametrize("interactive", [False, True])
def test_worker_start_failure_keeps_the_admitted_operation_receipt(
    tmp_path, clients, monkeypatch, interactive
):
    start_thread = Thread.start

    def start(thread):
        if thread.name.startswith("analysis-"):
            raise RuntimeError("no worker available")
        start_thread(thread)

    monkeypatch.setattr(Thread, "start", start)

    def respond(method, params):
        assert method == "tab.analyze"
        return _start_reply({}, [], interactive=interactive)

    client = _client(tmp_path, clients, respond)
    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    assert reply["isError"] is True
    stopped = json.loads(reply["content"][0]["text"])
    assert stopped["op"] == 1
    assert stopped["status"] == "failed"
    assert stopped["error"]["reason"] == "worker_start_failed"
    assert stopped["operation_outcome"] is None
    assert _methods(client) == ["tab.analyze"]


def test_gui_completion_during_initial_handoff_keeps_the_execution(
    tmp_path, clients, monkeypatch
):
    saved = Event()
    send_rpc = GuiConnection.send_gui_rpc

    def send(connection, method, params, *args, **kwargs):
        if method == "tab.interact":
            # The completion worker saves the figure before the initial interact read. The 30s limit only guards deadlock.
            assert saved.wait(30), "analysis did not continue before initial read"
        return send_rpc(connection, method, params, *args, **kwargs)

    monkeypatch.setattr(GuiConnection, "send_gui_rpc", send)

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [], interactive=True)
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {})
        if method == "tab.save_artifacts":
            return asdict(analysis_report("analysis", ["fit"]))
        if method == "tab.get_figure":
            saved.set()
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    client.transport.replies["tab.interact"] = {
        "ok": False,
        "error": {
            "code": "precondition_failed",
            "message": "interactive session already completed",
        },
    }
    started = _data(
        _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    )
    assert started["op"] == 1
    completed = _data(
        _call_full_execution_stdio(
            monkeypatch,
            client,
            "wait",
            # The released worker completes this execution; 30s only guards deadlock.
            {"execution": started["execution"], "timeout": 30},
        )
    )
    assert completed["status"] == "finished"
    assert completed["save_report"] == asdict(analysis_report("analysis", ["fit"]))
    assert _methods(client).count("tab.analyze") == 1
    assert _methods(client).count("tab.save_artifacts") == 1
