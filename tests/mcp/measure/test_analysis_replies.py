"""Captured analysis replies across admission and terminal outcomes."""

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.measure.analysis_execution import AnalysisWriteback
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.execution_reply import SummaryEstimate

import recipes

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
def test_analysis_captures_all_owner_writeback_without_live_queries(
    tmp_path, clients, stage
):
    pane = "analysis" if stage == "primary" else "post_analysis"
    proposal: AnalysisWriteback = {
        "has_draft": True,
        "items": [
            {
                "id": "candidate-t1",
                "kind": "metadict",
                "target_name": "t1",
                "proposed": 27.5,
                "current": 20.0,
                "selected": False,
            },
            {
                "id": "candidate-note",
                "kind": "sample",
                "target_name": "sample",
                "proposed": {"t1": 27.5, "source": "reanalysis"},
                "current": None,
                "selected": True,
            },
        ],
        "destination_context": {
            "active_label": "original",
            "chip_name": "chip",
            "qub_name": "q1",
            "res_name": "r1",
            "project_loaded": True,
        },
    }

    def respond(method, params):
        if method in {"tab.analyze", "tab.post_analyze"}:
            return _start_reply({"skip": 1}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method in {"tab.get_analyze_result", "tab.get_post_analyze_result"}:
            result = _result_reply(pane, [], {"skip": 1})
            result["summary"] = {"t1": 27.5}
            result["operation_state"][f"{pane}_state"]["has_writeback_draft"] = True
            return result
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond, writeback=proposal)
    initial = client.call("tab_analyze", {"tab": "t", "stage": stage})
    assert initial.data["status"] == "finished", initial.data
    execution = initial.data["execution"]
    expected_candidates = [
        {
            "id": "candidate-t1",
            "kind": "parameter",
            "target": "t1",
            "resolved_target": None,
            "proposed": 27.5,
            "current": 20.0,
        },
        {
            "id": "candidate-note",
            "kind": "sample",
            "target": "sample",
            "resolved_target": None,
            "proposed": {"t1": 27.5, "source": "reanalysis"},
            "current": None,
        },
    ]
    assert initial.data["writeback"]["stages"][stage] == expected_candidates
    assert (
        initial.data["writeback"]["stages"]["post" if stage == "primary" else "primary"]
        == []
    )
    assert initial.data["writeback"]["destination"] == {
        "context": {"active_label": "original"},
        "project": {"chip_name": "chip", "qub_name": "q1", "res_name": "r1"},
    }
    full = client.call("status", {"execution": execution, "detail": "full"})
    assert full["writeback"] == proposal
    assert full["writeback"]["items"][0]["selected"] is False
    assert [p for m, p in client.transport.sent if m == "tab.writeback_preview"] == [
        {"tab_id": "t", "subtab_id": pane, "operation_id": 71}
    ]
    proposal["items"][0]["proposed"] = 999.0
    proposal["destination_context"]["active_label"] = "newer"
    full["writeback"]["items"].clear()
    before = list(client.transport.sent)
    assert client.call("status", {"execution": execution}) == initial.data
    waited = client.call("wait", {"execution": execution, "timeout": 0}).data
    assert {k: v for k, v in waited.items() if k != "elapsed_s"} == initial.data
    assert (
        client.call("status", {"execution": execution, "detail": "full"})["writeback"][
            "items"
        ][0]["proposed"]
        == 27.5
    )
    assert client.transport.sent == before


@pytest.mark.parametrize(
    "declaration",
    [
        SummaryEstimate("other", "t1", "t1_err", "us"),
        SummaryEstimate("t1", "t1", "other_err", "us"),
        SummaryEstimate("t1", "t1", "t1_err", "ns"),
    ],
)
def test_analysis_only_rejects_conflicting_estimate_declarations(
    tmp_path, clients, monkeypatch, declaration
):
    t1 = next(recipe for recipe in recipes.RECIPES if recipe.name == "t1")
    conflict = replace(
        t1,
        name="conflicting-t1",
        summary_estimates=(declaration,),
    )
    monkeypatch.setattr(recipes, "RECIPES", (*recipes.RECIPES, conflict))

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            result = _result_reply("analysis", [], {})
            result["summary"] = {
                "t1": 25.0,
                "fit_quality": {
                    "fit": {
                        "r2": 0.9,
                        "normalized_residual_rms": 0.02,
                        "relative_parameter_errors": {},
                        "invalid": [],
                    }
                },
            }
            return result
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    with pytest.raises(ValueError, match="Ambiguous analysis estimate 't1'"):
        client.call("tab_analyze", {"tab": "t"})


@pytest.mark.parametrize("stage", ["primary", "post"])
def test_analysis_only_uses_unambiguous_native_quality_without_live_reads(
    tmp_path, clients, monkeypatch, stage
):
    t1 = next(recipe for recipe in recipes.RECIPES if recipe.name == "t1")
    identical = replace(t1, name="identical-t1")
    monkeypatch.setattr(recipes, "RECIPES", (*recipes.RECIPES, identical))
    native_issue = {
        "path": "summary.fit_quality.fit.relative_parameter_errors.decay_time",
        "reason": "covariance_unavailable",
    }
    quality = {
        "fit": {
            "r2": -0.25,
            "normalized_residual_rms": 0.31,
            "relative_parameter_errors": {"decay_time": None},
            "invalid": [native_issue],
        }
    }

    def respond(method, params):
        if method in {"tab.analyze", "tab.post_analyze"}:
            return _start_reply({}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method in {"tab.get_analyze_result", "tab.get_post_analyze_result"}:
            pane = "analysis" if stage == "primary" else "post_analysis"
            result = _result_reply(pane, [], {})
            result["summary"] = {"t1": 25.0, "t1_err": None, "fit_quality": quality}
            return result
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    initial = client.call("tab_analyze", {"tab": "t", "stage": stage})
    assert initial.data["status"] == "finished", initial.data
    execution = initial.data["execution"]
    before = len(client.transport.sent)
    summary = client.call("status", {"execution": execution})
    full = client.call("status", {"execution": execution, "detail": "full"})
    waited = client.call("wait", {"execution": execution, "timeout": 0}).data
    assert initial.data == summary
    assert {k: v for k, v in waited.items() if k != "elapsed_s"} == summary
    assert len(client.transport.sent) == before
    assert full["result"]["summary"]["fit_quality"] == quality
    estimate = summary["analysis"][stage]["estimates"]["t1"]
    assert estimate["value"] == 25.0
    assert estimate["unit"] == "us"
    assert estimate["quality"]["fit"]["r2"] == -0.25
    assert estimate["quality"]["fit"]["normalized_residual_rms"] == 0.31
    issue = {
        "path": f"analysis.{stage}.estimates.t1.quality.fit.relative_parameter_errors.decay_time",
        "reason": "covariance_unavailable",
    }
    assert estimate["quality"]["fit"]["invalid"] == [issue]
    assert summary["invalid"] == [issue]
    assert summary["analysis"][stage]["details"] == {}
    json.dumps(summary, allow_nan=False)
    json.dumps(full, allow_nan=False)


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
            "tab.writeback_preview",
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
    client.tools = build_measure_tools(
        client.context, recipes=client.context.session.recipes.definitions
    )
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
    assert _methods(client) == [
        method,
        "operation.await",
        result_method,
        "tab.writeback_preview",
        method,
    ]


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


@pytest.mark.parametrize("registered", [True, False], ids=["mcp-origin", "gui-origin"])
@pytest.mark.parametrize("outcome", ["finished", "failed"])
def test_done_joins_original_completion_without_duplicate_saves(
    tmp_path, clients, monkeypatch, registered, outcome
):
    done = Event()
    proposal: AnalysisWriteback = {
        "has_draft": False,
        "items": [],
        "destination_context": {"active_label": "done-destination"},
    }

    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({"gain": 2}, [], interactive=True)
        if method == "tab.interact":
            if "payload" in params:
                assert params["payload"] == {"command": "done"}
                assert params["include_figure"] is False
                proposal["has_draft"] = True
                proposal["items"] = [
                    {
                        "id": "final",
                        "kind": "metadict",
                        "target_name": "frequency",
                        "proposed": 8.5,
                        "current": 3.0,
                        "selected": False,
                    }
                ]
                done.set()
            return {
                "operation_id": 71,
                "state": {"value": 3},
                "commands": [{"name": "done"}],
                "figure": None,
            }
        if method == "operation.await":
            return (
                {
                    "reason": "completed",
                    "status": outcome,
                    "error": "fit failed" if outcome == "failed" else None,
                }
                if done.is_set()
                else {"reason": "user_feedback", "status": "running"}
            )
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {"gain": 2})
        if method == "tab.save_image":
            return {"image_path": "/actual/fit.png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond, writeback=proposal)
    started = (
        _data(
            _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
        )
        if registered
        else None
    )
    if started is not None:
        assert started["writeback"] is None
        assert "tab.writeback_preview" not in _methods(client)
    reply = _call_full_execution_stdio(
        monkeypatch,
        client,
        "tab_interact",
        {"tab": "t", "payload": {"command": "done"}},
    )
    assert bool(reply.get("isError")) is (outcome == "failed")
    completed = json.loads(reply["content"][0]["text"])
    assert completed["status"] == outcome
    assert completed["op"] == 1
    execution = completed["execution"]
    if started is not None:
        assert execution == started["execution"]
    terminal = _data(
        _call_full_execution_stdio(
            monkeypatch, client, "wait", {"execution": execution, "timeout": 0}
        )
    )
    assert terminal["status"] == outcome
    assert completed["save_status"] == (
        "saved" if outcome == "finished" else "not_started"
    )
    assert completed["saved_images"] == (
        [{"figure_name": "fit", "image_path": "/actual/fit.png"}]
        if outcome == "finished"
        else []
    )
    if outcome == "finished":
        assert completed["params"] == {"gain": 2}
        assert completed["writeback"] == proposal
        summary = client.call("status", {"execution": execution})
        assert summary["writeback"]["stages"]["primary"][0]["proposed"] == 8.5
        assert summary["writeback"]["destination"] == {
            "context": {"active_label": "done-destination"}
        }
        _assert_figure(reply, present=True)
    else:
        assert completed["error"]["reason"] == "analysis_failed"
    methods = _methods(client)
    assert methods.count("tab.analyze") == int(registered)
    assert methods.count("tab.interact") == 1 + int(registered)
    assert methods.count("tab.get_analyze_result") == int(outcome == "finished")
    assert methods.count("tab.save_image") == int(outcome == "finished")
    assert methods.count("tab.writeback_preview") == int(outcome == "finished")
    assert set(methods) <= {
        "tab.analyze",
        "tab.interact",
        "operation.await",
        "tab.get_analyze_result",
        "tab.save_image",
        "tab.get_figure",
        "tab.writeback_preview",
    }


@pytest.mark.parametrize("failure", ["writeback_rejected", "writeback_timeout"])
def test_failed_writeback_capture_keeps_saved_analysis_and_preview(
    tmp_path, clients, monkeypatch, failure
):
    def respond(method, params):
        if method == "tab.analyze":
            return _start_reply({"skip": 1}, [])
        if method == "operation.await":
            return {"reason": "completed", "status": "finished"}
        if method == "tab.get_analyze_result":
            return _result_reply("analysis", ["fit"], {"skip": 1})
        if method == "tab.save_image":
            return {"image_path": "/actual/fit.png"}
        if method == "tab.get_figure":
            return {"png_b64": base64.b64encode(_PNG).decode()}
        raise AssertionError(method)

    client = _client(tmp_path, clients, respond)
    _inject_analysis_rejection(client, failure, monkeypatch)
    reply = _call_full_execution_stdio(monkeypatch, client, "tab_analyze", {"tab": "t"})
    assert reply["isError"] is True
    data = json.loads(reply["content"][0]["text"])
    assert data["status"] == "failed"
    assert data["error"]["phase"] == "writeback_read"
    assert data["error"]["reason"] == (
        "superseded_result"
        if failure == "writeback_rejected"
        else "gui_handler_timeout"
    )
    assert data["writeback"] is None
    assert data["result"]["params"] == {"skip": 1}
    assert data["save_status"] == "saved"
    assert data["saved_images"] == [
        {"figure_name": "fit", "image_path": "/actual/fit.png"}
    ]
    assert Path(data["figure"]).read_bytes() == _PNG
    assert reply["content"][1]["data"] == base64.b64encode(_PNG).decode()
    assert _methods(client).count("tab.writeback_preview") == 1
    before = list(client.transport.sent)
    assert (
        client.call("status", {"execution": data["execution"], "detail": "full"})
        == data
    )
    assert client.transport.sent == before


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
