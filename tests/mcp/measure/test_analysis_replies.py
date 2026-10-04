"""Captured analysis replies across admission and terminal outcomes."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.measure.assembly import build_measure_tools

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
    assert_figure as _assert_figure,
)
from ._analyze_support import (
    call_full_execution_stdio as _call_full_execution_stdio,
)
from ._analyze_support import (
    hold_wire_reply as _hold_wire_reply,
)
from ._analyze_support import (
    inject_analysis_rejection as _inject_analysis_rejection,
)
from ._analyze_support import (
    sent_methods as _methods,
)
from ._analyze_support import (
    stdio_data as _data,
)


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("outcome", ["finished", "failed", "partial"])
def test_analysis_initial_wait_and_status_share_the_same_summary(
    tmp_path, clients, stage, outcome
):
    pane = "analysis" if stage == "primary" else "post_analysis"

    def respond(method, params):
        if method in {"tab.analyze", "tab.post_analyze"}:
            return _start_reply({"gain": 2.0}, [])
        if method == "operation.await":
            return {
                "reason": "completed",
                "status": "failed" if outcome == "failed" else "finished",
                "error": "fit failed" if outcome == "failed" else None,
            }
        if method in {"tab.get_analyze_result", "tab.get_post_analyze_result"}:
            return _result_reply(pane, ["fit", "residual"], {"gain": 2.0})
        if method == "tab.save_image":
            return {"image_path": "/actual/" + params["figure_name"] + ".png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    if outcome == "partial":

        def save(params):
            if params["figure_name"] == "residual":
                return {
                    "ok": False,
                    "error": {
                        "code": "precondition_failed",
                        "reason": "save_failed",
                        "message": "Destination unavailable",
                    },
                }
            return {"ok": True, "result": respond("tab.save_image", params)}

        client.transport.replies["tab.save_image"] = save
    initial = client.call("tab_analyze", {"tab": "t", "stage": stage})
    execution = initial.data["execution"]
    before = len(client.transport.sent)
    summary = client.call("status", {"execution": execution})
    full = client.call("status", {"execution": execution, "detail": "full"})
    waited = client.call("wait", {"execution": execution, "timeout": 0})
    assert initial.data == summary
    assert {k: v for k, v in waited.data.items() if k != "elapsed_s"} == summary
    assert len(client.transport.sent) == before
    assert summary["recipe"] is None
    assert summary["run_id"] is None
    assert summary["run_op"] is None
    assert summary["analysis"]["stage"] == stage
    assert summary["steps"]["analysis"][stage]["status"] == (
        "failed" if outcome == "failed" else "finished"
    )
    assert summary["status"] == ("finished" if outcome == "finished" else "failed")
    if outcome != "failed":
        assert summary["analysis"][stage]["params"] == {"gain": 2.0}
        assert summary["analysis"][stage]["details"] == {"frequency": 5.0}
        assert summary["artifacts"][pane]["fit"]["members"]["image"] == [
            {"path": "/actual/fit.png", "status": "saved"}
        ]
    else:
        assert summary["error"] == full["error"]
    if outcome == "partial":
        assert summary["steps"]["analysis_save"][stage]["status"] == "incomplete"
        assert summary["artifacts"][pane]["residual"] == {
            "status": "incomplete",
            "lifetime": "persistent",
            "members": {"image": []},
        }
        assert summary["error"]["phase"] == "image_save"


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize(
    "receipt, reason, step_status",
    [
        ("lost", "connection_lost", "unknown"),
        ("closed", "session_closed", "unknown"),
        ("timeout", "gui_transport_timeout", "unknown"),
        ("handler_timeout", "gui_handler_timeout", "unknown"),
        ("internal", "injected", "unknown"),
        ("stale", "stale_version", "not_started"),
        ("busy", "operation_busy", "not_started"),
    ],
)
def test_analysis_start_failure_retains_a_queryable_execution(
    tmp_path, clients, monkeypatch, stage, receipt, reason, step_status
):
    method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    client = _client(tmp_path, clients)
    pending = Event()
    if receipt in {"handler_timeout", "internal", "stale", "busy"}:
        client.transport.replies[method] = {
            "ok": False,
            "error": {
                "code": {
                    "handler_timeout": "timeout",
                    "internal": "internal",
                    "stale": "precondition_failed",
                    "busy": "busy",
                }[receipt],
                "reason": None if receipt == "handler_timeout" else reason,
                "message": "Analysis start did not provide a handle",
            },
        }
    elif receipt == "timeout":
        send_line = client.transport.send_line

        def send(payload):
            if payload["method"] != method:
                return send_line(payload)
            client.transport.sent.append((payload["method"], payload["params"]))
            raise GuiTransportTimeoutError(method, 0.01)

        monkeypatch.setattr(client.transport, "send_line", send)
    else:
        pending = _hold_wire_reply(client, monkeypatch, method)

    with ThreadPoolExecutor(max_workers=1) as pool:
        called = pool.submit(client.call, "tab_analyze", {"tab": "t", "stage": stage})
        if receipt in {"lost", "closed"}:
            assert pending.wait(1)
            if receipt == "closed":
                client.context.session.close()
            else:
                client.transport.close()
                assert client.transport.on_closed is not None
                client.transport.on_closed(None)
        initial = called.result(timeout=2)
    assert initial.is_error is True
    key = initial.data["execution"]
    before = len(client.transport.sent)
    summary = client.call("status", {"execution": key})
    full = client.call("status", {"execution": key, "detail": "full"})
    waited = client.call("wait", {"execution": key, "timeout": 0})
    assert summary == initial.data
    assert {k: v for k, v in waited.data.items() if k != "elapsed_s"} == summary
    assert summary["status"] == "failed"
    assert summary["steps"]["analysis"][stage]["status"] == step_status
    assert summary["error"]["reason"] == reason
    assert full["start"]["status"] == step_status
    assert full["op"] is None
    assert full["params"] is None
    assert summary["steps"]["analysis_save"][stage]["status"] == "not_started"
    assert summary["artifacts"]["analysis"] == {}
    assert summary["artifacts"]["post_analysis"] == {}
    assert len(client.transport.sent) == before
    assert _methods(client) == [method]


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("cancel", [False, True])
def test_pending_analysis_receipt_preserves_identity_and_cancel_intent(
    tmp_path, clients, stage, cancel
):
    method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    pane = "analysis" if stage == "primary" else "post_analysis"
    pending, release = Event(), Event()

    def respond(name, params):
        if name == method:
            pending.set()
            assert release.wait(2)
            return _start_reply({"gain": 2.0}, [])
        if name == "operation.cancel":
            return {"status": "cancelling"}
        if name == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if name in {"tab.get_analyze_result", "tab.get_post_analyze_result"}:
            return _result_reply(pane, [], {"gain": 2.0})
        raise AssertionError(name)

    client = _client(tmp_path, clients, respond)
    with ThreadPoolExecutor(max_workers=2) as pool:
        called = pool.submit(client.call, "tab_analyze", {"tab": "t", "stage": stage})
        try:
            assert pending.wait(1)
            snapshots = client.context.session.executions.snapshots()
            assert len(snapshots) == 1
            key = snapshots[0].execution
            before = len(client.transport.sent)
            summary = client.call("status", {"execution": key})
            full = client.call("status", {"execution": key, "detail": "full"})
            assert summary["op"] is None
            assert summary["steps"]["analysis"][stage]["status"] == "unknown"
            assert full["start"]["status"] == "unknown"
            assert len(client.transport.sent) == before
            if cancel:
                requested = pool.submit(client.call, "cancel", {"execution": key})
                stopped = requested.result(timeout=0.5)
                assert stopped.data["cancel_requested"] is True
                assert stopped.data["gui_cancel"]["status"] == "not_needed"
                assert len(client.transport.sent) == before
            release.set()
            initial = called.result(timeout=2)
            completed = client.call("wait", {"execution": key, "timeout": 2})
        finally:
            release.set()
    assert initial.data["execution"] == key
    assert completed.data["execution"] == key
    assert completed.data["status"] == ("cancelled" if cancel else "finished")
    assert completed.data["steps"]["analysis"][stage] == {
        "status": "finished",
        "reason": "completed",
    }
    full = client.call("status", {"execution": key, "detail": "full"})
    assert full["op"] == 1
    assert full["start"]["status"] == "running"
    assert full["cancel_requested"] == cancel
    assert full["operation_outcome"]["status"] == "finished"
    assert len(client.context.session.executions.snapshots()) == 1
    assert _methods(client) == (
        [method, "operation.cancel", "operation.await"]
        if cancel
        else [
            method,
            "operation.await",
            "tab.get_analyze_result"
            if stage == "primary"
            else "tab.get_post_analyze_result",
        ]
    )


@pytest.mark.parametrize("stage", ["primary", "post"])
def test_cancelled_analysis_queued_before_dispatch_never_starts(
    tmp_path, clients, stage, monkeypatch
):
    occupied, release, registered = Event(), Event(), Event()

    def respond(method, params):
        assert method == "tab.snapshot"
        occupied.set()
        assert release.wait(2)
        return {"tabs": []}

    client = _client(tmp_path, clients, respond)
    # A captured binding lets a second public tool register while RPC is occupied.
    client.context = client.context.bound()
    client.tools = build_measure_tools(client.context)
    register = client.context.session.executions.start

    def observe_registration(*args, **kwargs):
        execution = register(*args, **kwargs)
        registered.set()
        return execution

    # Synchronize the registry seam without replacing its admission logic.
    monkeypatch.setattr(
        client.context.session.executions, "start", observe_registration
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        blocker = pool.submit(
            client.call, "rpc_call", {"method": "tab.snapshot", "params": {}}
        )
        try:
            assert occupied.wait(1)
            called = pool.submit(
                client.call, "tab_analyze", {"tab": "t", "stage": stage}
            )
            assert registered.wait(1)
            snapshots = client.context.session.executions.snapshots()
            assert len(snapshots) == 1
            key = snapshots[0].execution
            before = len(client.transport.sent)
            summary = client.call("status", {"execution": key})
            assert summary["steps"]["analysis"][stage]["status"] == "not_started"
            stopped = client.call("cancel", {"execution": key})
            assert stopped.data["cancel_requested"] is True
            assert len(client.transport.sent) == before
            release.set()
            blocker.result(timeout=2)
            completed = called.result(timeout=2)
        finally:
            release.set()
    assert completed.data["execution"] == key
    assert completed.data["status"] == "cancelled"
    assert completed.data["steps"]["analysis"][stage]["status"] == "not_started"
    full = client.call("status", {"execution": key, "detail": "full"})
    assert full["start"]["status"] == "not_started"
    assert full["op"] is None
    assert full["operation_outcome"] is None
    assert full["error"] is None
    assert _methods(client) == ["tab.snapshot"]


@pytest.mark.parametrize("stage", ["primary", "post"])
def test_duplicate_analysis_receipt_retains_the_confirmed_handle(
    tmp_path, clients, stage
):
    method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    result_method = (
        "tab.get_analyze_result"
        if stage == "primary"
        else "tab.get_post_analyze_result"
    )
    pane = "analysis" if stage == "primary" else "post_analysis"

    def respond(name, params):
        if name == method:
            return _start_reply({}, [])
        if name == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if name == result_method:
            return _result_reply(pane, [], {})
        raise AssertionError(name)

    client = _client(tmp_path, clients, respond)
    first = client.call("tab_analyze", {"tab": "t", "stage": stage})
    conflicting = client.call("tab_analyze", {"tab": "t", "stage": stage})
    assert conflicting.is_error is True
    assert conflicting.data["execution"] != first.data["execution"]
    full = client.call(
        "status", {"execution": conflicting.data["execution"], "detail": "full"}
    )
    assert full["op"] == first.data["op"]
    assert full["start"]["status"] == "running"
    assert full["error"]["reason"] == "incompatible_wire"
    assert conflicting.data["steps"]["analysis"][stage]["status"] == "running"
    assert _methods(client) == [method, "operation.await", result_method, method]


@pytest.mark.parametrize("outcome", ["finished", "failed"])
def test_execution_query_observes_background_completion_without_reconnect(
    tmp_path, clients, monkeypatch, outcome
):
    release = Event()
    awaiting = Event()

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({"model": "fit"}, [])
        if method == "operation.await":
            awaiting.set()
            assert release.wait(10), "test did not release the GUI operation"
            return {"reason": "completed", "status": outcome, "error": None}
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {"model": "fit"})
        if method == "tab.save_image":
            return {"image_path": "/actual/fit.png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    try:
        started = _data(
            _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
        )
        assert awaiting.wait(2)
        assert started["status"] == "running"
        execution = started["execution"]
        running = _data(
            _call_full_execution_stdio(
                monkeypatch, client, "wait", {"execution": execution, "timeout": 0}
            )
        )
        assert running["status"] == "running"
        assert running["cancel_requested"] is False
        assert running["elapsed_s"] >= 0
        snapshot = client.call("status", {"execution": execution, "detail": "full"})
        snapshot["params"]["model"] = "caller mutation"
        assert client.call("status", {"execution": execution, "detail": "full"})[
            "params"
        ] == {"model": "fit"}
        assert _methods(client) == ["tab.analyze", "operation.await"]
        release.set()
        completed_reply = _call_full_execution_stdio(
            monkeypatch, client, "wait", {"execution": execution, "timeout": 2}
        )
        completed = _data(completed_reply)
        assert completed["status"] == outcome
        assert completed["operation_outcome"]["status"] == outcome
        assert completed["cancel_requested"] is False
        assert completed["save_status"] == (
            "saved" if outcome == "finished" else "not_started"
        )
        assert completed["saved_images"] == (
            [{"figure_name": "fit", "image_path": "/actual/fit.png"}]
            if outcome == "finished"
            else []
        )
        _assert_figure(completed_reply, present=outcome == "finished")
        sent = list(client.transport.sent)
        client.transport.close()
        terminal = _data(
            _call_full_execution_stdio(
                monkeypatch, client, "status", {"execution": execution}
            )
        )
        repeated = _data(
            _call_full_execution_stdio(
                monkeypatch, client, "wait", {"execution": execution, "timeout": 0}
            )
        )
        assert terminal == {
            key: value for key, value in repeated.items() if key != "elapsed_s"
        }
        assert terminal["status"] == outcome
        assert client.transport.sent == sent
    finally:
        release.set()


@pytest.mark.parametrize(
    "failure,phase,save_status,confirmed,unconfirmed",
    [
        ("result_rejected", "result_read", "not_started", [], None),
        ("save_rejected", "image_save", "incomplete", ["fit"], None),
        ("save_lost", "image_save", "unknown", ["fit"], "residual"),
        ("handler_timeout", "image_save", "unknown", ["fit"], "residual"),
        ("encoding_failed", "image_save", "unknown", ["fit"], "residual"),
        ("save_bad_path", "image_save", "unknown", ["fit"], "residual"),
        ("after_result_eof", "image_save", "incomplete", [], None),
        ("after_save_eof", "figure_read", "saved", ["fit", "residual"], None),
        ("preview_rejected", "figure_read", "saved", ["fit", "residual"], None),
        ("local_write", "figure_read", "saved", ["fit", "residual"], None),
    ],
)
def test_analysis_failure_retains_confirmed_prefix_without_replay(
    tmp_path,
    clients,
    monkeypatch,
    failure,
    phase,
    save_status,
    confirmed,
    unconfirmed,
):
    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({"gain": 2}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            if failure == "after_result_eof":
                client.transport.is_open = False
            return _result_reply("analysis", ["fit", "residual"], {"gain": 2})
        if method == "tab.save_image":
            if params["figure_name"] == "residual":
                if failure == "save_lost":
                    client.transport.is_open = False
                    raise OSError("socket closed during image save")
                if failure == "save_bad_path":
                    return {"image_path": None}
                if failure == "after_save_eof":
                    client.transport.is_open = False
            return {"image_path": f"/actual/{params['figure_name']}.png"}
        assert method == "tab.get_figure"
        return {"png_b64": base64.b64encode(_PNG).decode()}

    client = _client(tmp_path, clients, respond)
    _inject_analysis_rejection(client, failure, monkeypatch)

    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    assert reply["isError"] is True
    assert len(reply["content"]) == 1
    data = json.loads(reply["content"][0]["text"])
    assert data["status"] == "failed"
    assert data["error"]["phase"] == phase
    assert data["save_status"] == save_status
    assert data["saved_images"] == [
        {"figure_name": name, "image_path": f"/actual/{name}.png"} for name in confirmed
    ]
    assert data["unconfirmed_image"] == unconfirmed
    _assert_ambiguous_save_error(data, failure)
    if failure == "result_rejected":
        assert data["result"] is None
        assert data["remaining_images"] is None
    else:
        assert data["result"]["summary"] == {"frequency": 5.0}
        assert data["remaining_images"] == [
            name for name in ["fit", "residual"] if name not in confirmed
        ]
    assert data["figure"] is None
    methods = _methods(client)
    assert methods.count("tab.analyze") == 1
    assert methods.count("tab.get_analyze_result") == 1
    assert methods.count("tab.get_figure") == (
        1 if failure in ("preview_rejected", "local_write") else 0
    )
    assert [
        params["figure_name"]
        for method, params in client.transport.sent
        if method == "tab.save_image"
    ] == (
        []
        if failure in ("result_rejected", "after_result_eof")
        else ["fit", "residual"]
    )
