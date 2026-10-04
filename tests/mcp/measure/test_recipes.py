"""Recipe promises through the shipped tool table and GUI transport seam."""

import base64
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Any

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure import tools_recipes
from zcu_tools.mcp.measure.session import GuiRpcError, MeasureMcpSession

from ._recipe_support import PNG, LookbackGui
from ._support import full_execution_reply, make_client


@contextmanager
def recipe_client(tmp_path, respond):
    client = make_client(tmp_path, respond)
    try:
        yield client
    finally:
        client.context.session.close()


@pytest.mark.parametrize("outcome", ["finished", "missing", "failed"])
def test_recipe_initial_wait_and_status_share_the_same_summary(tmp_path, outcome):
    gui = LookbackGui()

    def respond(method, params):
        if outcome == "failed" and method == "operation.await":
            return {"reason": "completed", "status": "failed", "error": "Run failed"}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        initial = client.call(
            "lookback", {} if outcome == "missing" else {"frequency_mhz": 6020.0}
        )
        key = initial.data["execution"]
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": key})
        waited = client.call("wait", {"execution": key, "timeout": 0})
        full = client.call("status", {"execution": key, "detail": "full"})
        assert {k: v for k, v in initial.data.items() if k != "elapsed_s"} == summary
        assert {k: v for k, v in waited.data.items() if k != "elapsed_s"} == summary
        assert len(client.transport.sent) == before
        assert summary["status"] == (
            "needs_parameters" if outcome == "missing" else outcome
        )
        assert summary["run_id"] is None
        assert summary["steps"]["run"]["status"] == (
            "not_started" if outcome == "missing" else outcome
        )
        assert summary["previews"] == {
            "run": [],
            "primary": [full["analysis"]["figure"]] if outcome == "finished" else [],
            "post": [],
        }
        if outcome == "finished":
            assert initial.images
            assert waited.images
            assert summary["artifacts"]["raw"]["data"]["members"]["data"] == [
                {"path": "/actual/raw.h5", "status": "saved"}
            ]
            assert (
                full["actual"]["publication"]["cfg_ref"] == summary["actual"]["cfg_ref"]
            )
        elif outcome == "missing":
            assert summary["missing"] == full["missing"]
        else:
            assert summary["error"] == full["error"]


def test_writeback_destination_summary_keeps_context_and_project_identity(tmp_path):
    gui = LookbackGui()
    destination = {
        "active_label": "sample",
        "has_active_context": True,
        "chip_name": "chip",
        "qub_name": "q",
        "res_name": "r",
        "database_path": "/resolved/experiment/data",
        "result_dir": "/resolved/experiment/result",
    }

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.writeback_preview":
            response["destination_context"] = destination
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {"frequency_mhz": 6020.0})
        assert completed.data["status"] == "finished", completed.data
        key = completed.data["execution"]
        summary = client.call("status", {"execution": key})
        full = client.call("status", {"execution": key, "detail": "full"})
        assert summary["writeback"]["destination"] == {
            "context": {"active_label": "sample", "has_active_context": True},
            "project": {"chip_name": "chip", "qub_name": "q", "res_name": "r"},
        }
        assert full["writeback"]["destination_context"] == destination
        assert summary["artifacts"]["raw"]["data"]["members"]["data"] == [
            {"path": "/actual/raw.h5", "status": "saved"}
        ]


def test_module_candidate_summary_keeps_source_changes_and_full_proposal(tmp_path):
    gui = LookbackGui()
    current: dict[str, Any] = {
        "type": "pulse",
        "freq": 6100.0,
        "gain": 0.1,
        "phase": 0.0,
        "waveform": {"style": "const", "length": 0.1},
        "ch": 0,
        "nqz": 2,
        "pre_delay": 0.0,
        "post_delay": 0.0,
        "cloned_from": "calibrated_drive",
    }
    proposed: dict[str, Any] = {**deepcopy(current), "gain": 0.15}
    proposed["waveform"]["length"] = 0.24

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.writeback_preview":
            response["items"] = [
                {
                    "id": "ml-1",
                    "kind": "module",
                    "target_name": "pi_len",
                    "selected": False,
                    "proposed": proposed,
                    "current": current,
                }
            ]
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {"frequency_mhz": 6020.0})
        assert completed.data["status"] == "finished", completed.data
        key = completed.data["execution"]
        summary = client.call("status", {"execution": key})
        full = client.call("status", {"execution": key, "detail": "full"})
        candidate = summary["writeback"]["stages"]["primary"][0]
        assert candidate["kind"] == "module"
        assert candidate["target"] == "pi_len"
        assert candidate["resolved_target"] is None
        assert candidate["selected"] is False
        assert candidate["cfg_ref"] == summary["actual"]["cfg_ref"]
        assert candidate["proposed"] == {
            "type": "pulse",
            "freq": 6100.0,
            "gain": 0.15,
            "phase": 0.0,
            "waveform": {"style": "const", "length": 0.24},
            "cloned_from": "calibrated_drive",
        }
        assert candidate["current"]["cloned_from"] == "calibrated_drive"
        assert set(candidate["changes"]) == {"gain", "waveform.length"}
        assert full["writeback"]["items"][0]["proposed"] == proposed
        assert full["writeback"]["items"][0]["current"] == current


@pytest.mark.parametrize(
    "receipt, reason, step_status",
    [
        ("lost", "connection_lost", "unknown"),
        ("timeout", "gui_transport_timeout", "unknown"),
        ("handler_timeout", "gui_handler_timeout", "unknown"),
        ("internal", "injected", "unknown"),
        ("stale", "stale_version", "not_started"),
        ("busy", "operation_busy", "not_started"),
    ],
)
def test_run_start_failure_preserves_confirmed_rejection_or_ambiguity(
    tmp_path, monkeypatch, receipt, reason, step_status
):
    gui = LookbackGui()
    pending = Event()
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    with recipe_client(tmp_path, gui) as client:
        if receipt in {"handler_timeout", "internal", "stale", "busy"}:
            code = {
                "handler_timeout": "timeout",
                "internal": "internal",
                "stale": "precondition_failed",
                "busy": "busy",
            }[receipt]
            client.transport.replies["tab.run_start"] = {
                "ok": False,
                "error": {
                    "code": code,
                    "reason": None if receipt == "handler_timeout" else reason,
                    "message": "Run start did not provide a handle",
                },
            }
        else:
            send_line = client.transport.send_line

            def send(payload):
                if payload["method"] != "tab.run_start":
                    return send_line(payload)
                client.transport.sent.append((payload["method"], payload["params"]))
                gui("tab.run_start", payload["params"])
                pending.set()
                if receipt == "timeout":
                    raise GuiTransportTimeoutError("tab.run_start", 0.01)
                return None

            monkeypatch.setattr(client.transport, "send_line", send)
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        execution = initial.data["execution"]
        if receipt == "lost":
            assert pending.wait(1)
            client.transport.close()
            assert client.transport.on_closed is not None
            client.transport.on_closed(None)
        completed = client.call("wait", {"execution": execution, "timeout": 2})
        assert completed.data["status"] == "failed", completed.data
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert summary["steps"]["run"]["status"] == step_status
        assert full["run_start"]["status"] == step_status
        assert summary["error"]["reason"] == reason
        assert summary["run_op"] is None
        assert summary["actual"]["cfg_ref"] == full["actual"]["publication"]["cfg_ref"]
        assert summary["steps"]["raw_save"]["status"] == "not_started"
        assert len(client.transport.sent) == before
        assert [method for method, _ in client.transport.sent].count(
            "tab.run_start"
        ) == 1


def test_unconfirmed_run_receipt_stays_unknown_until_the_original_start_returns(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    pending = Event()
    release = Event()

    def respond(method, params):
        if method == "tab.run_start":
            pending.set()
            assert release.wait(2)
        return gui(method, params)

    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    with recipe_client(tmp_path, respond) as client:
        try:
            initial = client.call("lookback", {"frequency_mhz": 6020.0})
            assert pending.wait(1)
            execution = initial.data["execution"]
            before = len(client.transport.sent)
            summary = client.call("status", {"execution": execution})
            full = client.call("status", {"execution": execution, "detail": "full"})
            assert summary["steps"]["run"]["status"] == "unknown"
            assert summary["run_op"] is None
            assert full["run_start"]["status"] == "unknown"
            assert len(client.transport.sent) == before
            release.set()
            completed = client.call("wait", {"execution": execution, "timeout": 2})
            assert completed.data["status"] == "finished", completed.data
            confirmed = client.call("status", {"execution": execution})
            assert confirmed["steps"]["run"] == {
                "status": "finished",
                "reason": "completed",
            }
            assert confirmed["artifacts"]["raw"]["data"]["members"]["data"] == [
                {"path": "/actual/raw.h5", "status": "saved"}
            ]
            assert [method for method, _ in client.transport.sent].count(
                "tab.run_start"
            ) == 1
        finally:
            release.set()


def test_summary_status_reports_the_finished_recipe_facts(tmp_path):
    gui = LookbackGui()

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.writeback_preview":
            response["items"] = [
                {
                    "id": "md-1",
                    "kind": "metadict",
                    "target_name": "trigger_offset",
                    "selected": False,
                    "proposed": 0.24,
                    "current": 0.1,
                }
            ]
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {"frequency_mhz": 6020.0, "rounds": 7})
        assert completed.data["status"] == "finished", completed.data
        execution = completed.data["execution"]
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert summary["execution"] == full["execution"] == execution
        assert summary["run_id"] is None
        assert summary["actual"]["cfg_ref"] == full["actual"]["cfg_ref"]
        assert summary["actual"]["parameters"]["frequency_mhz"] == {
            "value": 6020.0,
            "source": "frequency_mhz",
            "unit": "MHz",
        }
        assert summary["actual"]["parameters"]["rounds"] == {
            "value": 7,
            "source": "rounds",
        }
        assert summary["actual"]["modules"]["reset"] == {
            "value": None,
            "source": "disabled",
        }
        assert summary["steps"]["run"] == {"status": "finished", "reason": "completed"}
        assert summary["steps"]["raw_save"]["status"] == "saved"
        assert summary["steps"]["analysis"]["primary"]["status"] == "finished"
        assert summary["steps"]["analysis_save"]["primary"]["status"] == "saved"
        assert summary["analysis"]["primary"]["params"] == {"threshold": 0.5}
        assert summary["analysis"]["primary"]["details"] == {"offset": 0.24}
        assert summary["artifacts"]["raw"]["data"] == {
            "status": "saved",
            "lifetime": "persistent",
            "members": {"data": [{"path": "/actual/raw.h5", "status": "saved"}]},
        }
        assert summary["artifacts"]["analysis"]["trace"]["members"] == {
            "image": [{"path": "/actual/trace.png", "status": "saved"}]
        }
        candidate = summary["writeback"]["stages"]["primary"][0]
        assert candidate == {
            "id": "md-1",
            "kind": "parameter",
            "target": "trigger_offset",
            "resolved_target": None,
            "proposed": 0.24,
            "current": 0.1,
            "selected": False,
        }
        assert summary["writeback"]["destination"] == {
            "context": {"active_label": "sample"}
        }
        assert summary["missing"] == summary["invalid"] == []
        assert summary["error"] is None
        assert len(client.transport.sent) == before


def test_full_query_keeps_the_publication_used_before_run(tmp_path):
    gui = LookbackGui()
    captured: dict[str, Any] = {}

    def respond(method, params):
        response = gui(method, params)
        if method == "tab.run_start":
            captured.update(deepcopy(gui.publication))
            gui.publication["cfg_ref"]["revision"] = "99"
            gui.publication["tree"]["children"]["rounds"] = {
                "kind": "scalar",
                "input": {"resolved": 999},
            }
        return response

    with recipe_client(tmp_path, respond) as client:
        completed = client.call("lookback", {"frequency_mhz": 6020.0, "rounds": 7})
        assert completed.data["status"] == "finished", completed.data
        execution = completed.data["execution"]
        before = len(client.transport.sent)
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert full["actual"]["publication"] == captured
        assert (
            full["actual"]["publication"]["tree"]["children"]["rounds"]["input"][
                "resolved"
            ]
            == 7
        )
        full["actual"]["publication"]["tree"]["children"].clear()
        repeated = client.call("status", {"execution": execution, "detail": "full"})
        assert repeated["actual"]["publication"] == captured
        assert len(client.transport.sent) == before


def test_lookback_initial_wait_returns_while_the_same_execution_continues(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    run_waiting = Event()
    release_run = Event()
    returned = Event()
    replies: list[ToolReply] = []

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            run_waiting.set()
            if not release_run.wait(0.02):
                return {"reason": "timeout"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)

    def call_recipe():
        replies.append(
            full_execution_reply(
                client, client.call("lookback", {"frequency_mhz": 6020.0})
            )
        )
        returned.set()

    caller = Thread(target=call_recipe)
    caller.start()
    try:
        assert run_waiting.wait(1)
        assert returned.wait(1), "Initial wait must not wait for Run completion"
        initial = replies[0].data
        assert initial["status"] == "running"
        execution = initial["execution"]
        initial["actual"]["fields"].clear()
        before = len(client.transport.sent)
        status = client.call("status", {"execution": execution, "detail": "full"})
        waiting = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 0})
        )
        assert status["actual"]["fields"]
        assert waiting.data["execution"] == execution
        assert waiting.data["status"] == "running"
        assert len(client.transport.sent) == before
        release_run.set()
        completed = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert completed.data["status"] == "finished", completed.data
        assert completed.data["run_op"] == initial["run_op"]
        assert completed.data["raw_save"]["path"] == "/actual/raw.h5"
        assert completed.images[0].data == PNG
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.analyze") == 1
    finally:
        release_run.set()
        caller.join(2)
        client.context.session.close()
    assert not caller.is_alive()


@pytest.fixture
def interactive_recipe(tmp_path, handoff_failure):
    gui = LookbackGui()
    done = Event()
    writeback_read = Event()
    initial_handoff = True

    def respond(method, params):
        nonlocal initial_handoff
        if method == "tab.analyze":
            return {**gui(method, params), "interactive": True}
        if (
            method == "operation.await"
            and params["operation_id"] == 93
            and not done.is_set()
        ):
            return {"reason": "user_feedback"}
        if method == "tab.interact":
            assert params["tab_id"] == "t"
            failure = handoff_failure if initial_handoff else None
            initial_handoff = False
            if params.get("payload", {}).get("command") == "done":
                done.set()
            return {
                "operation_id": 94 if failure == "replaced" else 93,
                "figure": {
                    "png_b64": "invalid"
                    if failure == "png"
                    else base64.b64encode(PNG).decode()
                }
                if params.get("include_figure", True)
                else None,
                "state": {"offset": 0.24},
                "commands": [{"name": "done"}],
                "prompt": "Confirm offset",
            }
        if method == "tab.writeback_preview":
            writeback_read.set()
        return gui(method, params)

    client = make_client(tmp_path, respond)
    if handoff_failure == "query":

        def reject_initial_query(params):
            nonlocal initial_handoff
            initial_handoff = False
            del client.transport.replies["tab.interact"]
            return {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "render_failed",
                    "message": "Preview unavailable",
                },
            }

        client.transport.replies["tab.interact"] = reject_initial_query
    try:
        yield client, writeback_read
    finally:
        done.set()
        client.context.session.close()


@pytest.mark.parametrize("handoff_failure", [None, "png", "query", "replaced"])
def test_lookback_interaction_handoff_keeps_the_original_pipeline_alive(
    interactive_recipe, handoff_failure
):
    client, writeback_read = interactive_recipe
    initial = client.call("lookback", {"frequency_mhz": 6020.0})
    handoff = full_execution_reply(client, initial)
    assert handoff.data["status"] == "interactive"
    assert handoff.data["raw_save"]["path"] == "/actual/raw.h5"
    assert handoff.data["analysis"]["op"] == handoff.data["op"]
    interaction = handoff.data["analysis"]["interaction"]
    assert interaction is not None
    if handoff_failure:
        assert handoff.is_error
        assert interaction["delivery_error"]
        assert interaction["figure"] is None
        assert not handoff.images
    else:
        assert not handoff.is_error
        assert handoff.images[0].data == PNG
        assert Path(interaction["figure"]).read_bytes() == PNG
    if handoff_failure not in ("query", "replaced"):
        assert interaction["state"] == {"offset": 0.24}
        assert interaction["commands"] == [{"name": "done"}]
    assert initial.data["previews"] == {
        "run": [],
        "primary": [] if handoff_failure else [interaction["figure"]],
        "post": [],
    }
    analysis_execution = handoff.data["analysis"]["execution"]
    execution = handoff.data["execution"]
    read = client.call("tab_interact", {"tab": "t"})
    assert read.data["prompt"] == "Confirm offset"
    assert read.images[0].data == PNG
    finished_analysis = client.call(
        "tab_interact", {"tab": "t", "payload": {"command": "done"}}
    )
    assert finished_analysis.data["status"] == "finished"
    assert writeback_read.wait(2), "Recipe must resume after interactive analysis"
    finished = full_execution_reply(
        client, client.call("wait", {"execution": execution, "timeout": 2})
    )
    assert finished.data["status"] == "finished", finished.data
    assert finished.data["analysis"]["execution"] == analysis_execution
    assert read.data["execution"] == analysis_execution
    assert finished.data["writeback"]["has_draft"]
    assert finished.images[0].data == PNG
    methods = [method for method, _ in client.transport.sent]
    assert methods.count("tab.run_start") == methods.count("tab.analyze") == 1


@pytest.mark.parametrize("held_op", [71, 82, 93])
def test_session_close_drains_recipe_work_and_rejects_new_admission(
    tmp_path, monkeypatch, held_op
):
    gui = LookbackGui()
    pending = Event()
    disconnected = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == held_op:
            pending.set()
            assert disconnected.wait(2), "Session must disconnect before joining"
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    close_transport = client.transport.close

    def disconnect():
        close_transport()
        disconnected.set()

    monkeypatch.setattr(client.transport, "close", disconnect)
    try:
        initial = client.call("lookback", {"frequency_mhz": 6020.0})
        assert pending.wait(1)
        client.context.session.close()
        before = len(client.transport.sent)
        result = client.call(
            "status", {"execution": initial.data["execution"], "detail": "full"}
        )
        assert result["status"] == "failed", result
        assert result["phase"] == "terminal"
        assert result["error"]["reason"] in ("session_closed", "connection_lost")
        if held_op == 82:
            assert result["raw_save"]["status"] == "unknown"
            assert result["raw_save"]["reserved_path"] == "/actual/raw.h5"
            assert result["raw_save"]["path"] is None
        with pytest.raises(GuiRpcError, match="closed"):
            client.call("lookback", {"frequency_mhz": 6020.0})
        client.context.session.close()
        assert len(client.transport.sent) == before
    finally:
        disconnected.set()
        client.context.session.close()


def test_lookback_cancel_latches_one_control_and_stops_after_original_run(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    stopped = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            if stopped.is_set():
                return {"reason": "completed", "status": "cancelled"}
            return {"reason": "timeout"}
        if method == "operation.cancel":
            assert params == {"operation_id": 71}
            stopped.set()
            return {"status": "cancelling"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        execution = initial.data["execution"]
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["execution"] == execution
        client.call("cancel", {"op": initial.data["run_op"]})
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["run_outcome"]["status"] == "cancelled"
        before = client.call("status", {"execution": execution})
        client.call("cancel", {"execution": execution})
        assert client.call("status", {"execution": execution}) == before
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.run_start") == 1
        assert "tab.save_data" not in methods
        assert "tab.analyze" not in methods
    finally:
        stopped.set()
        client.context.session.close()


@pytest.mark.parametrize(
    ("partial_available", "cancel_after", "expected"),
    [(True, False, "finished"), (False, False, "failed"), (True, True, "cancelled")],
)
def test_lookback_finish_early_uses_partial_data_unless_cancel_wins(
    tmp_path, monkeypatch, partial_available, cancel_after, expected
):
    gui = LookbackGui()
    release_run = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            return (
                {"reason": "completed", "status": "cancelled"}
                if release_run.is_set()
                else {"reason": "timeout"}
            )
        if method == "operation.cancel":
            assert params == {"operation_id": 71}
            return {"status": "cancelling"}
        reply = gui(method, params)
        if method == "tab.snapshot" and gui.ran:
            reply["tabs"][0]["result_state"]["available"] = partial_available
        return reply

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        execution = initial.data["execution"]
        early = client.call("finish_early", {"op": initial.data["run_op"]})
        assert early.data == {
            "execution": execution,
            "op": initial.data["op"],
            "run_op": initial.data["run_op"],
            "status": "running",
            "phase": "run",
            "cancel_requested": False,
            "finish_early_requested": True,
            "gui_cancel": {"status": "requested", "error": None},
        }
        client.call("finish_early", {"execution": execution})
        if cancel_after:
            client.call("cancel", {"execution": execution})
        release_run.set()
        result = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert result.data["status"] == expected, result.data
        assert result.data["run_outcome"]["status"] == "cancelled"
        frozen = client.call("status", {"execution": execution, "detail": "full"})
        later = client.call("finish_early", {"execution": execution})
        assert later.data == {
            "execution": execution,
            "op": result.data["op"],
            "run_op": initial.data["run_op"],
            "status": "not_applicable",
            "phase": "terminal",
            "cancel_requested": cancel_after,
            "finish_early_requested": True,
            "gui_cancel": None,
        }
        assert (
            client.call("status", {"execution": execution, "detail": "full"}) == frozen
        )
        if expected == "finished":
            assert result.data["raw_save"]["path"] == "/actual/raw.h5"
            assert result.data["analysis"]["status"] == "finished"
        elif expected == "failed":
            assert result.data["error"]["reason"] == "run_result_unavailable"
            assert result.data["raw_save"]["status"] == "not_started"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.save_data") == int(expected == "finished")
        assert methods.count("tab.analyze") == int(expected == "finished")
    finally:
        release_run.set()
        client.context.session.close()


@pytest.mark.parametrize("save_succeeds", [True, False])
def test_lookback_cancel_during_raw_save_waits_for_the_true_save_outcome(
    tmp_path, monkeypatch, save_succeeds
):
    gui = LookbackGui()
    saving = Event()
    release_save = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 82:
            saving.set()
            if not release_save.is_set():
                return {"reason": "timeout"}
            if not save_succeeds:
                return {"reason": "completed", "status": "failed", "error": "disk full"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        assert saving.wait(1)
        execution = initial.data["execution"]
        early = client.call("finish_early", {"execution": execution})
        assert early.data["status"] == "not_applicable"
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["gui_cancel"]["status"] == "not_cancellable"
        assert cancelled.data["cancel_requested"]
        during_save = client.call("status", {"execution": execution, "detail": "full"})
        assert during_save["raw_save"]["status"] == "saving"
        assert during_save["raw_save"]["path"] is None
        release_save.set()
        final = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert not final.is_error  # Successful query, even when execution failed.
        assert final.data["status"] == ("cancelled" if save_succeeds else "failed")
        assert final.data["raw_save"]["status"] == (
            "saved" if save_succeeds else "failed"
        )
        assert final.data["raw_save"]["path"] == (
            "/actual/raw.h5" if save_succeeds else None
        )
        if not save_succeeds:
            assert final.data["error"]["reason"] == "raw_save_failed"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.save_data") == 1
        assert "operation.cancel" not in methods
        assert "tab.analyze" not in methods
    finally:
        release_save.set()
        client.context.session.close()


@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_recipe_cancel_before_initial_handoff_joins_the_true_analysis_outcome(
    tmp_path, monkeypatch, outcome
):
    gui = LookbackGui()
    handoff_waiting = Event()
    release_handoff = Event()
    allow_terminal = Event()
    original_send = MeasureMcpSession.GuiConnection.send_gui_rpc

    def delay_handoff(self, method, params, *args, **kwargs):
        # Schedule the public connection call before its real admission check.
        if method == "tab.interact":
            handoff_waiting.set()
            assert release_handoff.wait(2)
        return original_send(self, method, params, *args, **kwargs)

    def respond(method, params):
        if method == "tab.analyze":
            return {**gui(method, params), "interactive": True}
        if method == "operation.await" and params["operation_id"] == 93:
            if allow_terminal.is_set():
                return {
                    "reason": "completed",
                    "status": outcome,
                    "error": "Analysis failed after cancellation",
                }
            return {"reason": "user_feedback"}
        if method == "operation.cancel":
            assert params == {"operation_id": 93}
            return {"status": "cancelling"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(MeasureMcpSession.GuiConnection, "send_gui_rpc", delay_handoff)
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        assert handoff_waiting.wait(1)
        execution = initial.data["execution"]
        (analysis_receipt,) = client.context.session.executions.snapshots()
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["gui_cancel"]["status"] == "requested"
        release_handoff.set()
        allow_terminal.set()
        terminal = initial
        for _ in range(100):
            terminal = full_execution_reply(
                client, client.call("wait", {"execution": execution, "timeout": 0.01})
            )
            if terminal.data["phase"] == "terminal":
                break
            sleep(0.01)
        assert terminal.data["status"] == outcome
        assert terminal.data["analysis"]["execution"] == analysis_receipt.execution
        assert terminal.data["analysis"]["op"] == analysis_receipt.op
        assert terminal.data["analysis"]["status"] == outcome
        assert terminal.data["analysis"]["cancel_requested"]
        assert terminal.data["analysis"]["operation_outcome"]["status"] == outcome
        assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
        if outcome == "failed":
            assert terminal.data["error"]["reason"] == "analysis_failed"
        methods = [method for method, _ in client.transport.sent]
        assert all(
            methods.count(method) == 1
            for method in ("tab.run_start", "tab.analyze", "operation.cancel")
        )
        assert not {
            "tab.interact",
            "tab.get_analyze_result",
            "tab.save_image",
            "tab.get_figure",
            "tab.writeback_preview",
        }.intersection(methods)
    finally:
        release_handoff.set()
        allow_terminal.set()
        client.context.session.close()


@pytest.mark.parametrize("selector", ["execution", "run_op", "op"])
def test_recipe_cancel_delegates_to_the_existing_analysis_owner(
    tmp_path, monkeypatch, selector
):
    gui = LookbackGui()
    analyzing = Event()
    stopped = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 93:
            analyzing.set()
            return (
                {"reason": "completed", "status": "cancelled"}
                if stopped.is_set()
                else {"reason": "timeout"}
            )
        if method == "operation.cancel":
            assert params == {"operation_id": 93}
            stopped.set()
            return {"status": "cancelling"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        assert analyzing.wait(1)
        execution = initial.data["execution"]
        state = client.call("status", {"execution": execution})
        key = "execution" if selector == "execution" else "op"
        arguments = {key: state[selector]}
        early = client.call("finish_early", arguments)
        assert early.data["status"] == "not_applicable"
        assert not stopped.is_set()
        cancelled = client.call("cancel", arguments)
        assert cancelled.data["cancel_requested"]
        assert cancelled.data["gui_cancel"]["status"] == "requested"
        client.call("cancel", {"execution": execution})
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["analysis"]["status"] == "cancelled"
        assert terminal.data["analysis"]["cancel_requested"]
        assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.analyze") == 1
        assert "tab.writeback_preview" not in methods
    finally:
        stopped.set()
        client.context.session.close()


@pytest.mark.parametrize(
    ("pending_method", "forbidden_method"),
    [
        ("tab.snapshot", "tab.save_data"),
        ("operation.await", "tab.analyze"),
        ("tab.get_figure", "tab.writeback_preview"),
    ],
)
def test_recipe_cancel_blocks_the_next_phase_while_an_admitted_read_finishes(
    tmp_path, monkeypatch, pending_method, forbidden_method
):
    gui = LookbackGui()
    pending = Event()
    release = Event()
    control_replies = []

    def respond(method, params):
        at_boundary = method == pending_method and (
            (method == "tab.snapshot" and gui.ran)
            or (method == "operation.await" and params["operation_id"] == 82)
            or method == "tab.get_figure"
        )
        if at_boundary:
            pending.set()
            assert release.wait(2)
        if method == "operation.cancel":
            return {"status": "finished"}
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    controller = None
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        assert pending.wait(1)
        execution = initial.data["execution"]
        controller = Thread(
            target=lambda: control_replies.append(
                client.call("cancel", {"execution": execution})
            )
        )
        controller.start()
        state = client.call("status", {"execution": execution})
        for _ in range(100):
            state = client.call("status", {"execution": execution})
            if state["cancel_requested"]:
                break
            release.wait(0.01)
        assert state["cancel_requested"]
        assert forbidden_method not in [method for method, _ in client.transport.sent]
        release.set()
        controller.join(2)
        assert not controller.is_alive()
        assert control_replies[0].data["cancel_requested"]
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert terminal.data["status"] == "cancelled"
        methods = [method for method, _ in client.transport.sent]
        assert forbidden_method not in methods
        assert methods.count("tab.run_start") == 1
        if pending_method != "tab.snapshot":
            assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
    finally:
        release.set()
        if controller is not None:
            controller.join(2)
        client.context.session.close()


@pytest.mark.parametrize(
    ("method_pending", "wire_op", "control"),
    [
        ("tab.run_start", 71, "cancel"),
        ("tab.run_start", 71, "finish_early"),
        ("tab.analyze", 93, "cancel"),
    ],
)
def test_recipe_control_reaches_the_original_operation_after_a_late_receipt(
    tmp_path, monkeypatch, method_pending, wire_op, control
):
    gui = LookbackGui()
    pending = Event()
    release = Event()
    stopped = Event()

    def respond(method, params):
        if method == method_pending:
            pending.set()
            assert release.wait(2)
        if method == "operation.cancel":
            assert params == {"operation_id": wire_op}
            stopped.set()
            return {"status": "cancelling"}
        if method == "operation.await" and params["operation_id"] == wire_op:
            return (
                {"reason": "completed", "status": "cancelled"}
                if stopped.is_set()
                else {"reason": "timeout"}
            )
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        assert pending.wait(1)
        execution = initial.data["execution"]
        for _ in range(2):
            reply = client.call(control, {"execution": execution})
            assert reply.data[f"{control}_requested"]
        assert not stopped.is_set()
        release.set()
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert stopped.is_set()
        assert terminal.data["status"] == (
            "finished" if control == "finish_early" else "cancelled"
        )
        methods = [method for method, _ in client.transport.sent]
        assert methods.count(method_pending) == 1
        assert methods.count("operation.cancel") == 1
        if method_pending == "tab.analyze":
            assert terminal.data["analysis"]["cancel_requested"]
            assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
            assert "tab.get_analyze_result" not in methods
        elif control == "cancel":
            assert "tab.save_data" not in methods
        else:
            assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
    finally:
        release.set()
        stopped.set()
        client.context.session.close()


def test_recipe_cancel_during_preparation_wins_over_a_missing_parameter_handoff(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    pending = Event()
    release = Event()

    def respond(method, params):
        if method == "tab.edit_cfg":
            pending.set()
            assert release.wait(2)
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = client.call("lookback", {})
        assert pending.wait(1)
        execution = initial.data["execution"]
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        release.set()
        terminal = client.call("wait", {"execution": execution, "timeout": 2})
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["cancel_requested"]
        assert terminal.data["tab"] == "t"
        assert terminal.data["missing"] == []
        assert not gui.ran
    finally:
        release.set()
        client.context.session.close()


def test_recipe_worker_start_failure_returns_a_terminal_receipt_and_closes_safely(
    tmp_path, monkeypatch
):
    def fail_start(self):
        raise RuntimeError("no thread resources")

    with recipe_client(tmp_path, LookbackGui()) as client:
        monkeypatch.setattr(Thread, "start", fail_start)
        reply = client.call("lookback", {"frequency_mhz": 6020.0})
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["phase"] == "preparing"
        assert reply.data["error"]["reason"] == "worker_start_failed"
        assert reply.data["error"]["message"] == "no thread resources"
        execution = reply.data["execution"]
        state = client.call("status", {"execution": execution})
        assert state["status"] == "failed"
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["status"] == "failed"
        assert not cancelled.data["cancel_requested"]
        assert "tab.new" not in [method for method, _ in client.transport.sent]


def test_recipe_cancel_during_admitted_writeback_preserves_result_and_intent(
    tmp_path, monkeypatch
):
    gui = LookbackGui()
    reading = Event()
    release = Event()

    def respond(method, params):
        if method == "tab.writeback_preview":
            reading.set()
            assert release.wait(2)
        return gui(method, params)

    client = make_client(tmp_path, respond)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        assert reading.wait(1)
        execution = initial.data["execution"]
        cancelled = client.call("cancel", {"execution": execution})
        assert cancelled.data["cancel_requested"]
        release.set()
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert terminal.data["status"] == "cancelled"
        assert terminal.data["cancel_requested"]
        assert terminal.data["writeback"]["items"][0]["proposed"] == 0.24
        assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
        assert terminal.data["analysis"]["status"] == "finished"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.writeback_preview") == 1
        assert "operation.cancel" not in methods
    finally:
        release.set()
        client.context.session.close()


@pytest.mark.parametrize("replacement_source", [99, None])
@pytest.mark.parametrize("available", [False, True])
def test_lookback_rejects_a_post_run_snapshot_from_another_source(
    tmp_path, replacement_source, available
):
    gui = LookbackGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.snapshot" and gui.ran:
            reply["tabs"][0]["result_state"] = {
                "available": available,
                "revision": 99,
                "source_operation_id": replacement_source,
            }
        return reply

    with recipe_client(tmp_path, respond) as client:
        reply = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6020.0})
        )
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "result_superseded"
        assert reply.data["run_outcome"]["status"] == "finished"
        assert reply.data["result_state"] is None
        assert reply.data["raw_save"]["status"] == "not_started"
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert "tab.save_data" not in methods
        assert "tab.analyze" not in methods


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize(
    "readout_length,offset,expected",
    [(4.0, 0.2, (4.0, 0.2)), (1, 1, (1.0, 1.0)), (1.0, 1.0, (1.0, 1.0))],
)
def test_lookback_saves_original_run_then_analysis_and_delivers_complete_reply(
    tmp_path, reuse, readout_length, offset, expected
):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "lookback",
                {
                    "reuse_tab_id": "t" if reuse else None,
                    "frequency_mhz": 6020.0,
                    "readout_length_us": readout_length,
                    "trigger_offset_us": offset,
                    "rounds": 7,
                },
            ),
        )
        assert isinstance(reply, ToolReply)
        data = reply.data
        assert data["status"] == "finished", data
        assert not reply.is_error
        assert data["tab"] == "t"
        assert data["run_op"] != 71
        assert data["run_outcome"]["status"] == "finished"
        assert data["raw_save"]["path"] == "/actual/raw.h5"
        assert data["analysis"]["result"]["summary"] == {"offset": 0.24}
        assert data["analysis"]["saved_images"] == [
            {"figure_name": "trace", "image_path": "/actual/trace.png"}
        ]
        assert data["writeback"]["items"] == [{"id": "md-1", "proposed": 0.24}]
        assert reply.images[0].data == PNG
        assert data["elapsed_s"] >= 0
        actual = data["actual"]
        assert actual["cfg_ref"] == gui.publication["cfg_ref"]
        fields = actual["fields"]
        for path, value in {
            "modules.readout.pulse_cfg.freq": 6020.0,
            "modules.readout.ro_cfg.ro_freq": 6020.0,
            "modules.readout.ro_cfg.ro_length": expected[0],
            "modules.readout.ro_cfg.trig_offset": expected[1],
            "rounds": 7,
        }.items():
            assert fields[path]["value"] == value
            assert type(fields[path]["value"]) is type(value)
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.reset_cfg") == int(reuse)
        assert methods.count("tab.new") == int(not reuse)
        assert methods.count("tab.save_data") == 1
        assert methods.count("tab.analyze") == 1
        assert methods.index("device.snapshot") < methods.index("tab.run_start")
        assert methods.index("soc.info") < methods.index("tab.run_start")
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
        assert methods.index("tab.save_image") < methods.index("tab.writeback_preview")
        edits = [
            edit
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
            for edit in params["edits"]
        ]
        by_path = {tuple(edit["path"]): edit["value"] for edit in edits}
        assert by_path["modules", "reset"] == {"__ref": None}
        assert by_path["modules", "init_pulse"] == {"__ref": None}
        for key, value in zip(("ro_length", "trig_offset"), expected, strict=True):
            assert by_path["modules", "readout", "ro_cfg", key] == value
            assert type(by_path["modules", "readout", "ro_cfg", key]) is float


@pytest.mark.parametrize("reuse_tab_id", [None, "kept"])
def test_lookback_missing_frequency_does_not_run_a_blind_default(
    tmp_path, reuse_tab_id
):
    tab = reuse_tab_id or "new"
    publication = {
        "cfg_ref": {"cfg_id": "cfg", "revision": "2"},
        "status": "Valid",
        "source_basis": [],
        "diagnostics": [],
        "tree": {
            "kind": "section",
            "valid": True,
            "children": {
                "modules": {
                    "kind": "section",
                    "valid": True,
                    "children": {
                        "readout": {
                            "kind": "reference",
                            "ref": None,
                            "valid": True,
                            "children": {},
                        },
                    },
                },
            },
        },
    }

    def respond(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "context.snapshot":
            return {"label": "sample", "md": {}, "ml": {"modules": {}, "waveforms": {}}}
        if method == "tab.new":
            return {"tab_id": tab}
        if method == "tab.snapshot":
            return {
                "tabs": [
                    {
                        "tab_id": tab,
                        "adapter_name": "lookback",
                        "interaction": {
                            "is_running": False,
                            "is_analyzing": False,
                            "is_saving_data": False,
                        },
                    }
                ]
            }
        if method in ("tab.get_cfg", "tab.reset_cfg", "tab.edit_cfg"):
            return deepcopy(publication)
        raise AssertionError(f"Missing frequency must not start work: {method}")

    with recipe_client(tmp_path, respond) as client:
        arguments = {} if reuse_tab_id is None else {"reuse_tab_id": reuse_tab_id}
        reply = client.call("lookback", arguments)
        assert isinstance(reply, ToolReply)
        assert reply.data["status"] == "needs_parameters"
        assert reply.data["tab"] == tab
        assert [item["parameter"] for item in reply.data["missing"]] == [
            "frequency_mhz"
        ]
        assert reply.is_error is False
        methods = [method for method, _ in client.transport.sent]
        assert "tab.run_start" not in methods
        assert ("tab.new" in methods) is (reuse_tab_id is None)
        assert ("tab.reset_cfg" in methods) is (reuse_tab_id is not None)


@pytest.mark.parametrize(
    "source", ["r_f", "library", "library_error", "library_invalid"]
)
def test_lookback_uses_only_valid_frequency_sources_and_keeps_gui_defaults(
    tmp_path, source
):
    gui = LookbackGui()
    gui.md["r_f"] = 6500.0
    readout = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    pulse = readout["children"]["pulse_cfg"]["children"]["freq"]
    adc = readout["children"]["ro_cfg"]["children"]["ro_freq"]
    pulse["input"]["resolved"] = 6100.0
    adc["input"]["resolved"] = 6110.0
    arguments = {} if source == "r_f" else {"readout_ref": "calibrated"}
    if source == "library_error":
        pulse["input"]["error"] = "unresolved"
    if source == "library_invalid":
        pulse["valid"] = False
        pulse["input"]["validation_error"] = "outside range"
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(client, client.call("lookback", arguments))
        assert reply.data["status"] == "finished", reply.data
        fields = reply.data["actual"]["fields"]
        expected_pulse = 6100.0 if source == "library" else 6500.0
        expected_adc = 6500.0 if source == "r_f" else 6110.0
        assert fields["modules.readout.pulse_cfg.freq"]["value"] == expected_pulse
        assert fields["modules.readout.ro_cfg.ro_freq"]["value"] == expected_adc
        assert fields["modules.readout.pulse_cfg.freq"]["source"] == (
            "library:calibrated" if source == "library" else "r_f"
        )
        assert fields["modules.readout.ro_cfg.ro_length"]["value"] == 2.0
        assert fields["modules.readout.ro_cfg.trig_offset"]["value"] == 0.1
        assert fields["rounds"]["value"] == 3


@pytest.mark.parametrize(
    "arguments",
    [
        {"frequency_mhz": True},
        {"frequency_mhz": float("nan")},
        {"readout_length_us": float("inf")},
        {"trigger_offset_us": "0.1"},
        {"rounds": True},
        {"rounds": 1.0},
        {"rounds": 2.5},
        {"reuse_tab_id": ""},
        {"readout_ref": ""},
        {"use_reset": False},
        {"init_pulse_ref": 1},
    ],
)
def test_lookback_invalid_explicit_inputs_never_fall_back_or_create_tab(
    tmp_path, arguments
):
    gui = LookbackGui()
    gui.md["r_f"] = 6500.0
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("lookback", arguments)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["tab"] is None
        assert not gui.ran
        assert not any(method.startswith("tab.") for method, _ in client.transport.sent)


@pytest.mark.parametrize("partial_data", [False, True])
def test_manual_gui_run_cancel_never_starts_the_recipe_save_pipeline(
    tmp_path, partial_data
):
    gui = LookbackGui()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            gui.ran = partial_data
            return {"reason": "completed", "status": "cancelled"}
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        reply = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6000.0})
        )
        assert reply.data["status"] == "cancelled"
        assert not reply.data["cancel_requested"]
        assert not reply.data["finish_early_requested"]
        assert reply.data["run_outcome"]["status"] == "cancelled"
        assert reply.data["raw_save"]["status"] == "not_started"
        methods = [method for method, _ in client.transport.sent]
        assert "tab.save_data" not in methods
        assert "tab.analyze" not in methods
        assert "operation.cancel" not in methods


@pytest.mark.parametrize(
    ("argument", "module"),
    [
        ("readout_ref", "readout"),
        ("use_reset", "reset"),
        ("init_pulse_ref", "init_pulse"),
    ],
)
def test_explicit_invalid_library_reference_is_not_disabled_or_replaced(
    tmp_path, argument, module
):
    gui = LookbackGui()
    gui.md["r_f"] = 6500.0
    gui.publication["tree"]["children"]["modules"]["children"][module].update(
        valid=False, error="missing library entry"
    )
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("lookback", {argument: "missing"})
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "invalid_cfg"
        assert reply.data["tab"] == "t"
        edits = [
            params
            for method, params in client.transport.sent
            if method == "tab.edit_cfg"
        ]
        assert len(edits) == 1
        assert {"path": ["modules", module], "value": {"__ref": "missing"}} in edits[0][
            "edits"
        ]
        assert "tab.run_start" not in [method for method, _ in client.transport.sent]


@pytest.mark.parametrize(
    ("pending_method", "wire_op", "phase", "raw_status"),
    [
        ("operation.await", 71, "run", "not_started"),
        ("tab.save_data", None, "raw_save", "unknown"),
        ("operation.await", 82, "raw_save", "unknown"),
        ("operation.await", 93, "analysis", "saved"),
    ],
)
def test_recipe_connection_loss_retains_known_prefix_without_reconnecting(
    tmp_path, monkeypatch, pending_method, wire_op, phase, raw_status
):
    client = make_client(tmp_path, LookbackGui(), port_is_open=lambda port: True)
    send_line = client.transport.send_line
    pending = Event()
    reconnects = []

    def send(payload):
        if payload["method"] == pending_method and (
            wire_op is None or payload["params"]["operation_id"] == wire_op
        ):
            client.transport.sent.append((payload["method"], payload["params"]))
            pending.set()
            return
        send_line(payload)

    def unexpected_connect(*args, **kwargs):
        reconnects.append((args, kwargs))
        raise AssertionError("Recipe must not reconnect")

    monkeypatch.setattr(client.transport, "send_line", send)
    monkeypatch.setattr(client.context.bridge, "connect", unexpected_connect)
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    try:
        initial = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6000.0})
        )
        assert pending.wait(1)
        client.transport.close()
        assert client.transport.on_closed is not None
        client.transport.on_closed(None)
        execution = initial.data["execution"]
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 2})
        )
        assert terminal.data["status"] == "failed"
        assert terminal.data["error"]["reason"] == "connection_lost"
        assert terminal.data["error"]["phase"] == phase
        assert terminal.data["raw_save"]["status"] == raw_status
        if wire_op == 82:
            assert terminal.data["raw_save"]["reserved_path"] == "/actual/raw.h5"
            assert terminal.data["raw_save"]["path"] is None
        if raw_status == "saved":
            assert terminal.data["raw_save"]["path"] == "/actual/raw.h5"
        state = client.call("status", {"execution": execution})
        assert state["error"] == terminal.data["error"]
        assert not reconnects
        assert client.transport.sent[-1][0] == pending_method
        assert [method for method, _ in client.transport.sent].count(
            "tab.run_start"
        ) == 1
    finally:
        client.context.session.close()


@pytest.mark.parametrize("failure", ["wrong_experiment", "busy", "missing", "stale"])
def test_lookback_reuse_errors_do_not_create_replacement_or_retry(tmp_path, failure):
    gui = LookbackGui()
    client = make_client(tmp_path, gui)
    if failure == "stale":
        client.transport.replies["tab.reset_cfg"] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "stale_cfg",
                "message": "changed",
            },
        }
    else:
        snapshot = gui("tab.snapshot", {"tab_id": "t"})
        if failure == "missing":
            snapshot["tabs"] = []
        elif failure == "busy":
            snapshot["tabs"][0]["interaction"]["is_analyzing"] = True
        else:
            snapshot["tabs"][0]["adapter_name"] = "other"
        client.transport.replies["tab.snapshot"] = {"ok": True, "result": snapshot}
    try:
        reply = client.call("lookback", {"reuse_tab_id": "t", "frequency_mhz": 6000.0})
        assert reply.is_error
        assert reply.data["tab"] == "t"
        methods = [method for method, _ in client.transport.sent]
        assert "tab.new" not in methods
        assert "tab.run_start" not in methods
        assert methods.count("tab.reset_cfg") == int(failure == "stale")
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "failure, phase, raw_status",
    [
        ("run", "run", "not_started"),
        ("raw_start", "raw_save", "failed"),
        ("raw_finish", "raw_save", "failed"),
        ("analysis", "analysis", "saved"),
        ("superseded_raw", "raw_save", "failed"),
        ("superseded_analysis", "analysis", "saved"),
        ("superseded_writeback", "writeback_read", "saved"),
        ("image", "analysis", "saved"),
        ("writeback", "writeback_read", "saved"),
    ],
)
def test_lookback_failure_preserves_completed_prefix_without_retry(
    tmp_path, failure, phase, raw_status
):
    gui = LookbackGui()

    def respond(method, params):
        fail_operation = {"run": 71, "raw_finish": 82, "analysis": 93}.get(failure)
        if method == "operation.await" and params["operation_id"] == fail_operation:
            return {
                "reason": "completed",
                "status": "failed",
                "error": "injected failure",
            }
        return gui(method, params)

    client = make_client(tmp_path, respond)
    failed_method = {
        "raw_start": "tab.save_data",
        "superseded_raw": "tab.save_data",
        "superseded_analysis": "tab.analyze",
        "superseded_writeback": "tab.writeback_preview",
        "image": "tab.save_image",
        "writeback": "tab.writeback_preview",
    }.get(failure)
    if failed_method:
        client.transport.replies[failed_method] = {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "result_superseded"
                if failure.startswith("superseded")
                else "injected",
                "message": "injected failure",
            },
        }
    try:
        reply = full_execution_reply(
            client, client.call("lookback", {"frequency_mhz": 6000.0})
        )
        data = reply.data
        assert data["status"] == "failed", data
        assert reply.is_error
        assert data["error"]["phase"] == phase
        if failure.startswith("superseded"):
            assert data["error"]["reason"] == "result_superseded"
        assert data["raw_save"]["status"] == raw_status
        assert data["tab"] == "t"
        assert data["run_op"] is not None
        if raw_status == "saved":
            assert data["raw_save"]["path"] == "/actual/raw.h5"
        if failure == "raw_finish":
            assert data["raw_save"]["reserved_path"] == "/actual/raw.h5"
            assert data["raw_save"]["operation_outcome"]["status"] == "failed"
        if failure in ("image", "writeback"):
            assert data["analysis"]["result"]["summary"] == {"offset": 0.24}
        if failure == "writeback":
            assert data["analysis"]["saved_images"] == [
                {"figure_name": "trace", "image_path": "/actual/trace.png"}
            ]
            assert reply.images[0].data == PNG
        methods = [method for method, _ in client.transport.sent]
        for method in (
            "tab.run_start",
            "tab.save_data",
            "tab.analyze",
            "tab.writeback_preview",
        ):
            assert methods.count(method) <= 1
        if raw_status != "saved":
            assert "tab.analyze" not in methods
    finally:
        client.context.session.close()
