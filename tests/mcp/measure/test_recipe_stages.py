"""Same-Run primary/post stage ownership with an injected recording pipeline."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from threading import Event

import pytest
from zcu_tools.mcp.core.bridge import GuiTransportTimeoutError
from zcu_tools.mcp.measure.raw_save import SaveOutput
from zcu_tools.mcp.measure.recipe import RecipeAnalysis, RecipeRun, RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_stage_support import (
    POST_REPORT,
    PRIMARY_REPORT,
    RAW_REPORT,
    StageGui,
    skip_writeback,
    start_pipeline,
)
from ._recipe_support import PNG
from ._recipe_sync_support import cancel_while_rpc_held, wait_for_event
from ._support import full_execution_reply, make_client


@pytest.fixture()
def stage_client(tmp_path):
    gui = StageGui()
    client = make_client(tmp_path, gui)
    try:
        yield gui, client
    finally:
        client.context.session.close()


@pytest.fixture()
def background_stastage_client(stage_client):
    return stage_client


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("receipt", ["delayed", "lost", "timeout"])
def test_unconfirmed_analysis_start_retains_unknown_and_saved_prefix(
    background_stastage_client, monkeypatch, stage, receipt
):
    gui, client = background_stastage_client
    pending, release = Event(), Event()
    method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    send_line = client.transport.send_line

    def send(payload):
        if payload["method"] != method:
            return send_line(payload)
        if receipt == "delayed":
            pending.set()
            wait_for_event(release, "Test must release the admitted RPC")
            return send_line(payload)
        client.transport.sent.append((payload["method"], payload["params"]))
        gui(method, payload["params"])
        pending.set()
        if receipt == "timeout":
            raise GuiTransportTimeoutError(method, 0.01)
        return None

    monkeypatch.setattr(client.transport, "send_line", send)
    try:
        initial = start_pipeline(client, pending)
        execution = initial.data["execution"]
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert summary["steps"]["analysis"][stage]["status"] == "unknown"
        assert full["analysis_starts"][stage]["status"] == "unknown"
        assert (
            summary["artifacts"]["raw"]["data"]["members"]["data"]
            == asdict(RAW_REPORT)["artifacts"]["raw"]["data"]["members"]["data"]
        )
        if stage == "post":
            assert summary["artifacts"]["analysis"]["results"]["status"] == "Saved"
        assert len(client.transport.sent) == before
        if receipt == "delayed":
            release.set()
        elif receipt == "lost":
            client.transport.close()
            assert client.transport.on_closed is not None
            client.transport.on_closed(None)
        # Released work must reach a handoff; timeout only guards deadlock.
        completed = client.call("wait", {"execution": execution, "timeout": 30})
        confirmed = client.call("status", {"execution": execution})
        if receipt == "delayed":
            assert completed.data["status"] == "awaiting_answer", completed.data
            skip_writeback(client, completed)
            assert confirmed["steps"]["analysis"][stage]["status"] == "finished"
        else:
            assert completed.data["status"] == "failed", completed.data
            assert confirmed["steps"]["analysis"][stage]["status"] == "unknown"
            assert confirmed["error"]["reason"] == (
                "connection_lost" if receipt == "lost" else "gui_transport_timeout"
            )
        assert [name for name, _ in client.transport.sent].count(method) == 1
    finally:
        release.set()


@pytest.mark.parametrize("stage", ["primary", "post"])
def test_cancel_during_writeback_retains_stage_and_prevents_next_admission(
    background_stastage_client, stage
):
    gui, client = background_stastage_client
    pending, release = Event(), Event()
    operation = 93 if stage == "primary" else 104

    def writeback(params):
        if params["operation_id"] == operation:
            pending.set()
            wait_for_event(release, "Test must release the admitted RPC")
        return {"ok": True, "result": gui("tab.writeback_preview", params)}

    client.transport.replies["tab.writeback_preview"] = writeback
    client.transport.replies["operation.cancel"] = {
        "ok": True,
        "result": {"status": "finished"},
    }
    try:
        initial = full_execution_reply(client, start_pipeline(client, pending))
        execution = initial.data["execution"]
        status = client.call("status", {"execution": execution, "detail": "full"})
        assert status["analysis_stage"] == stage
        cancelled = cancel_while_rpc_held(client, execution, release)
        assert cancelled.data["gui_cancel"]["status"] == "not_needed"
        # RPC released, cancel caller joined: worker can finish; 30s guards deadlock.
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 30})
        )
        data = terminal.data
        assert data["status"] == "cancelled", data
        assert data["writeback"]["items"][0]["id"] == "param-1"
        assert data["analysis"]["status"] == (
            "cancelled" if stage == "primary" else "finished"
        )
        assert data["raw_save"] == asdict(RAW_REPORT)
        assert len(terminal.images) == (1 if stage == "primary" else 2)
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.post_analyze") == (0 if stage == "primary" else 1)
        assert methods.count("operation.cancel") == 1
        if stage == "post":
            assert data["post_writeback"]["items"][0]["id"] == "classifier"
    finally:
        release.set()


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize("outcome", ["cancelled", "failed"])
def test_cancel_targets_late_stage_receipt_and_joins_true_outcome(
    background_stastage_client, stage, outcome
):
    gui, client = background_stastage_client
    pending, release, stopped = Event(), Event(), Event()
    start_method = "tab.analyze" if stage == "primary" else "tab.post_analyze"
    operation = 93 if stage == "primary" else 104

    def start(params):
        pending.set()
        wait_for_event(release, "Test must release the admitted RPC")
        return {"ok": True, "result": gui(start_method, params)}

    def cancel(params):
        assert params == {"operation_id": operation}
        stopped.set()
        return {"ok": True, "result": {"status": "cancelling"}}

    def await_operation(params):
        if params["operation_id"] != operation:
            return {"ok": True, "result": gui("operation.await", params)}
        result = (
            {"reason": "completed", "status": outcome, "error": "failure after cancel"}
            if stopped.is_set()
            else {"reason": "timeout"}
        )
        return {"ok": True, "result": result}

    client.transport.replies.update(
        {
            start_method: start,
            "operation.cancel": cancel,
            "operation.await": await_operation,
        }
    )
    try:
        initial = full_execution_reply(client, start_pipeline(client, pending))
        execution = initial.data["execution"]
        for _ in range(2):
            assert client.call("cancel", {"execution": execution}).data[
                "cancel_requested"
            ]
        assert not stopped.is_set()
        release.set()
        terminal = full_execution_reply(
            # Released work must reach a handoff; timeout only guards deadlock.
            client,
            client.call("wait", {"execution": execution, "timeout": 30}),
        )
        data = terminal.data
        assert stopped.is_set()
        assert data["status"] == outcome, data
        current = data["analysis" if stage == "primary" else "post_analysis"]
        assert current["status"] == outcome
        assert current["cancel_requested"]
        assert current["operation_outcome"]["status"] == outcome
        assert data["raw_save"] == asdict(RAW_REPORT)
        assert len(terminal.images) == (1 if stage == "post" else 0)
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("operation.cancel") == 1
        assert methods.count("tab.post_analyze") == (1 if stage == "post" else 0)
        assert methods.count("tab.get_post_analyze_result") == 0
        if stage == "post":
            assert data["analysis"]["status"] == "finished"
            assert data["writeback"]["items"][0]["id"] == "param-1"
    finally:
        release.set()
        stopped.set()


@pytest.mark.parametrize("save_failed", [False, True])
def test_cancel_during_post_save_preserves_real_save_outcome(
    background_stastage_client, save_failed
):
    gui, client = background_stastage_client
    pending, release = Event(), Event()

    def save(params):
        if params["operation_id"] == 104:
            pending.set()
            wait_for_event(release, "Test must release the admitted RPC")
            if save_failed:
                return {
                    "ok": False,
                    "error": {
                        "code": "io_error",
                        "reason": "save_failed",
                        "message": "save failed after cancellation",
                    },
                }
        return {"ok": True, "result": gui("tab.save_artifacts", params)}

    client.transport.replies["tab.save_artifacts"] = save
    client.transport.replies["operation.cancel"] = {
        "ok": True,
        "result": {"status": "finished"},
    }
    try:
        initial = full_execution_reply(client, start_pipeline(client, pending))
        execution = initial.data["execution"]
        cancelled = cancel_while_rpc_held(client, execution, release)
        assert cancelled.data["gui_cancel"]["status"] == "not_needed"
        # RPC released, cancel caller joined: worker can finish; 30s guards deadlock.
        terminal = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 30})
        )
        data = terminal.data
        assert data["status"] == ("failed" if save_failed else "cancelled"), data
        assert data["post_analysis"]["save_status"] == (
            "incomplete" if save_failed else "saved"
        )
        assert data["post_analysis"]["save_report"] == (
            None if save_failed else asdict(POST_REPORT)
        )
        assert data["analysis"]["status"] == "finished"
        assert data["raw_save"] == asdict(RAW_REPORT)
        assert len(terminal.images) == 1
        assert data["post_writeback"] is None
        assert not any(
            method in ("tab.get_figure", "tab.writeback_preview")
            and params.get("operation_id") == 104
            for method, params in client.transport.sent
        )
    finally:
        release.set()


@pytest.mark.parametrize(
    "method_pending,operation",
    [
        ("tab.writeback_preview", 93),
        ("tab.post_analyze", None),
        pytest.param("tab.save_artifacts", 104, id="tab.save_image-104"),
    ],
)
def test_close_stops_stage_admission_without_reconnect(
    background_stastage_client, monkeypatch, method_pending, operation
):
    gui, client = background_stastage_client
    pending, disconnected = Event(), Event()
    original_close = client.transport.close

    def respond(params):
        if operation is None or params.get("operation_id") == operation:
            pending.set()
            wait_for_event(disconnected, "Test must release the admitted RPC")
        return {"ok": True, "result": gui(method_pending, params)}

    def disconnect():
        original_close()
        disconnected.set()

    client.transport.replies[method_pending] = respond
    monkeypatch.setattr(client.transport, "close", disconnect)
    try:
        initial = start_pipeline(client, pending)
        before = len(client.transport.sent)
        client.context.session.close()
        data = client.call(
            "status", {"execution": initial.data["execution"], "detail": "full"}
        )
        assert data["status"] == "cancelled", data
        assert data["phase"] == "terminal"
        assert data["error"] is None
        assert data["raw_save"] == asdict(RAW_REPORT)
        assert data["analysis"]["status"] == (
            "failed" if method_pending == "tab.writeback_preview" else "finished"
        )
        if method_pending == "tab.writeback_preview":
            assert data["analysis"]["error"]["phase"] == "writeback_read"
            assert data["analysis"]["writeback"] is None
        assert data["analysis"]["save_report"] == asdict(PRIMARY_REPORT)
        assert len(client.transport.sent) == before
        client.context.session.close()
    finally:
        disconnected.set()


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize(
    "failure", ["start", "analysis", "result", "save", "png", "writeback"]
)
def test_stage_failure_preserves_completed_prefix(stage_client, stage, failure):
    gui, client = stage_client
    operation = 93 if stage == "primary" else 104
    result_method = (
        "tab.get_analyze_result"
        if stage == "primary"
        else "tab.get_post_analyze_result"
    )
    failing_method = {
        "start": "tab.analyze" if stage == "primary" else "tab.post_analyze",
        "analysis": "operation.await",
        "result": result_method,
        "save": "tab.save_artifacts",
        "png": "tab.get_figure",
        "writeback": "tab.writeback_preview",
    }[failure]

    def result_reply(params):
        result = gui(result_method, params)
        if stage == "post":
            result["operation_state"]["post_analysis_state"]["figure_names"] = [
                "cloud",
                "histogram",
            ]
        return {"ok": True, "result": result}

    def failure_reply(params):
        is_target = failure == "start" or params.get("operation_id") == operation
        if not is_target:
            return {"ok": True, "result": gui(failing_method, params)}
        if failure == "analysis":
            return {
                "ok": True,
                "result": {
                    "reason": "completed",
                    "status": "failed",
                    "error": "analysis failed",
                },
            }
        return {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "reason": "result_superseded",
                "message": "injected failure",
            },
        }

    client.transport.replies[result_method] = result_reply
    client.transport.replies[failing_method] = failure_reply
    reply = full_execution_reply(
        client, client.call("pipeline", {"frequency_mhz": 5000.0, "post": True})
    )
    data = reply.data
    assert data["status"] == "failed", data
    assert data["tab"] == "t"
    assert data["raw_save"] == asdict(RAW_REPORT)
    assert data["analysis_stage"] == stage
    methods = [method for method, _ in client.transport.sent]
    assert methods.count("tab.run_start") == 1
    assert methods.count("tab.post_analyze") == (1 if stage == "post" else 0)
    assert data["post_writeback"] is None
    if stage == "post":
        assert data["analysis"]["status"] == "finished"
        assert data["analysis"]["save_report"] == asdict(PRIMARY_REPORT)
        assert data["writeback"]["items"][0]["id"] == "param-1"
    else:
        assert data["writeback"] is None
        assert data["post_analysis"] is None
    current = data["analysis" if stage == "primary" else "post_analysis"]
    if failure == "start":
        assert current["status"] == "failed"
        assert current["start"] == {
            "status": "not_started",
            "reason": "result_superseded",
        }
        assert current["op"] is None
    elif failure == "save":
        assert current["save_report"] is None
    elif failure in ("png", "writeback"):
        assert current["save_report"] == (
            asdict(POST_REPORT) if stage == "post" else asdict(PRIMARY_REPORT)
        )
    assert len(reply.images) == (1 if stage == "post" else 0) + (
        1 if failure == "writeback" else 0
    )


def test_preserves_invalid_analysis_and_saved_paths_without_accepting(stage_client):
    gui, client = stage_client
    for method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):

        def result(params, result_method=method):
            observed = gui(result_method, params)
            observed["summary"]["stderr"] = None
            observed["invalid"] = [{"path": "summary.stderr", "reason": "non_finite"}]
            return {"ok": True, "result": observed}

        client.transport.replies[method] = result
    reply = full_execution_reply(
        client, client.call("pipeline", {"frequency_mhz": 5000.0, "post": True})
    )
    reply = full_execution_reply(client, skip_writeback(client, reply))
    data = json.loads(json.dumps(reply.data, allow_nan=False))
    assert data["status"] == "finished"
    assert data["raw_save"] == asdict(RAW_REPORT)
    for stage, summary, image in (
        ("analysis", {"offset": 0.24, "stderr": None}, "trace"),
        ("post_analysis", {"fidelity": 0.98, "stderr": None}, "cloud"),
    ):
        assert data[stage]["result"]["summary"] == summary
        assert data[stage]["result"]["invalid"] == [
            {"path": "summary.stderr", "reason": "non_finite"}
        ]
        assert data[stage]["save_report"] == (
            asdict(PRIMARY_REPORT) if image == "trace" else asdict(POST_REPORT)
        )
    assert "tab.writeback_apply" not in [method for method, _ in client.transport.sent]


@pytest.mark.parametrize("decision", ["accepted", "skipped"])
@pytest.mark.parametrize("interactive", [False, True])
def test_writeback_waits_for_both_stages_and_reports_actual_receipts(
    tmp_path, decision, interactive
):
    gui = StageGui(interactive=interactive)
    client = make_client(tmp_path, gui)
    try:
        reply = client.call("pipeline", {"frequency_mhz": 5000.0, "post": True})
        execution = reply.data["execution"]
        if interactive:
            assert reply.data["status"] == "interactive"
            post = client.call(
                "tab_interact", {"tab": "t", "payload": {"command": "done"}}
            )
            assert post.data["execution"] == execution
            assert post.data["status"] == "interactive"
            assert post.data["analysis"]["stage"] == "post"
            reply = client.call(
                "tab_interact", {"tab": "t", "payload": {"command": "done"}}
            )
        assert reply.data["status"] == "awaiting_answer", reply.data
        assert reply.data["question_items"] == ("predict_offset", "classifier")
        assert reply.data["writeback"]["receipts"] == ()
        assert len(reply.images) == 2
        assert [image.data for image in reply.images] == [PNG, PNG]
        assert gui.writes == []
        before = list(client.transport.sent)
        summary = client.call("status", {"execution": execution})
        full = client.call("status", {"execution": execution, "detail": "full"})
        assert summary["question_items"] == full["question_items"]
        assert client.transport.sent == before
        result = client.call("answer", {"recipe": execution, "decision": decision})
        assert result.data["status"] == "finished", result.data
        assert len(result.images) == 2
        receipts = result.data["writeback"]["receipts"]
        if decision == "accepted":
            assert [request["subtab_id"] for request in gui.writes] == [
                "analysis",
                "post_analysis",
            ]
            assert [request["write"] for request in gui.writes] == [
                [{"id": "param-1"}],
                [{"id": "classifier"}],
            ]
            assert len(receipts) == 1
            assert [stage["stage"] for stage in receipts[0]["completed"]] == [
                "primary",
                "post",
            ]
        else:
            assert gui.writes == [] and receipts == ()
    finally:
        client.context.session.close()


def test_answer_failure_keeps_confirmed_primary_write_and_all_previews(stage_client):
    gui, client = stage_client
    question = client.call("pipeline", {"frequency_mhz": 5000.0, "post": True})
    assert question.data["status"] == "awaiting_answer", question.data

    def write(params):
        if params["subtab_id"] == "post_analysis":
            return {
                "ok": False,
                "error": {
                    "code": "io_error",
                    "reason": "write_failed",
                    "message": "Post write failed",
                },
            }
        return {"ok": True, "result": gui("tab.writeback_write", params)}

    client.transport.replies["tab.writeback_write"] = write
    failed = client.call(
        "answer", {"recipe": question.data["execution"], "decision": "accepted"}
    )
    assert failed.data["status"] == "failed" and failed.is_error
    assert len(failed.images) == 2
    (receipt,) = failed.data["writeback"]["receipts"]
    assert [stage["stage"] for stage in receipt["completed"]] == ["primary"]
    assert receipt["failed_stage"] == "post"
    assert receipt["failed_stage_may_have_partial_writes"]
    assert len(gui.writes) == 1
    full = client.call(
        "status", {"execution": failed.data["execution"], "detail": "full"}
    )
    assert full["raw_save"] == asdict(RAW_REPORT)
    assert full["analysis"]["status"] == full["post_analysis"]["status"] == "finished"


def test_saves_and_delivers_primary_then_post_without_rerun(tmp_path):
    gui = StageGui()

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.get_analyze_result":
            reply["summary"] = {
                "centers": {"ground": [0.1, 0.2], "excited": [0.8, 0.9]}
            }
        elif method == "tab.get_post_analyze_result":
            reply["summary"] = {
                "fidelity": 0.98,
                "populations": {"ground": 0.97, "excited": 0.03},
            }
        return reply

    client = make_client(tmp_path, respond)
    try:
        reply = full_execution_reply(
            client,
            skip_writeback(
                client,
                client.call("pipeline", {"frequency_mhz": 5000.0, "post": True}),
            ),
        )
        data = reply.data
        assert data["status"] == "finished", data
        assert data["analysis_mode"] == "primary_post"
        assert data["analysis_stage"] == "post"
        assert data["analysis"]["result"]["summary"] == {
            "centers": {"ground": [0.1, 0.2], "excited": [0.8, 0.9]}
        }
        assert data["post_analysis"]["result"]["summary"] == {
            "fidelity": 0.98,
            "populations": {"ground": 0.97, "excited": 0.03},
        }
        before = len(client.transport.sent)
        summary = client.call("status", {"execution": data["execution"]})
        assert len(client.transport.sent) == before
        assert summary["analysis"]["primary"]["summary"] == {
            "centers": {"ground": [0.1, 0.2], "excited": [0.8, 0.9]}
        }
        assert summary["analysis"]["post"]["summary"] == {
            "fidelity": 0.98,
            "populations": {"ground": 0.97, "excited": 0.03},
        }
        assert summary["artifacts"] == {
            **asdict(RAW_REPORT)["artifacts"],
            **asdict(PRIMARY_REPORT)["artifacts"],
            **asdict(POST_REPORT)["artifacts"],
        }
        assert data["analysis"]["save_report"] == asdict(PRIMARY_REPORT)
        assert data["post_analysis"]["save_report"] == asdict(POST_REPORT)
        assert data["writeback"]["items"][0]["id"] == "param-1"
        assert data["post_writeback"]["items"][0]["id"] == "classifier"
        assert len(reply.images) == 2
        assert gui.publication["input"]["modules"]["reset"] is None
        assert gui.publication["input"]["modules"]["init_pulse"] is None
        stages = {
            "tab.run_start",
            "tab.save_data",
            "tab.analyze",
            "tab.post_analyze",
            "tab.save_artifacts",
            "tab.get_figure",
            "tab.writeback_preview",
        }
        assert [method for method, _ in gui.calls if method in stages] == [
            "tab.run_start",
            "tab.save_data",
            "tab.analyze",
            "tab.save_artifacts",
            "tab.get_figure",
            "tab.writeback_preview",
            "tab.post_analyze",
            "tab.save_artifacts",
            "tab.get_figure",
            "tab.writeback_preview",
        ]
    finally:
        client.context.session.close()


@pytest.mark.parametrize("outcome", ["saved", "partial", "transport", "png"])
def test_author_post_capture_retains_reports_images_and_original_sources(
    tmp_path, outcome
):
    gui = StageGui()
    client = make_client(tmp_path, gui)
    try:
        run, status = (
            RecipeSession(client.context)
            .open_tab("lookback")
            .run()
            .complete_in_current_worker()
        )
        assert isinstance(run, RecipeRun) and status == "completed"
        assert run.save_raw() == RAW_REPORT
        primary, status = run.analyze("primary").complete_in_current_worker()
        assert isinstance(primary, RecipeAnalysis) and status == "completed"
        assert primary.snapshot.save_report == PRIMARY_REPORT
        expected = deepcopy(POST_REPORT)
        if outcome == "partial":
            artifact = expected.artifacts["post_analysis"]["results"]
            artifact.members["figure:cloud"].append(
                SaveOutput("png", "/copies/cloud.png", "failed", "copy unavailable")
            )
            expected.artifacts["post_analysis"]["results"] = replace(
                artifact, status="Partially saved"
            )

        def save(params):
            assert params == {
                "tab_id": "t",
                "artifacts": [{"section": "post_analysis", "name": "results"}],
                "operation_id": 104,
            }
            if outcome == "transport":
                return {
                    "ok": False,
                    "error": {
                        "code": "precondition_failed",
                        "reason": "result_superseded",
                        "message": "source replaced",
                    },
                }
            return {"ok": True, "result": asdict(expected)}

        if outcome in ("partial", "transport"):
            client.transport.replies["tab.save_artifacts"] = save
        if outcome == "png":
            client.transport.replies["tab.get_figure"] = {
                "ok": False,
                "error": {
                    "code": "precondition_failed",
                    "reason": "result_superseded",
                    "message": "source replaced",
                },
            }
        post = run.analyze("post")
        if outcome in ("transport", "png"):
            with pytest.raises(GuiRpcError) as failure:
                post.complete_in_current_worker()
            assert failure.value.reason == "result_superseded"
        else:
            result, status = post.complete_in_current_worker()
            assert isinstance(result, RecipeAnalysis) and status == "completed"
        capture = post.snapshot()
        assert capture is not None
        assert capture.status == (
            "failed" if outcome in ("transport", "png") else "finished"
        )
        assert capture.result is not None
        assert capture.result.summary == {"fidelity": 0.98}
        assert capture.save_report == (None if outcome == "transport" else expected)
        assert capture.save_status == (
            "incomplete" if outcome in ("partial", "transport") else "saved"
        )
        assert [image.data for image in post.preview_images()] == (
            [] if outcome in ("transport", "png") else [PNG]
        )
        assert (capture.writeback is None) == (outcome in ("transport", "png"))
        assert run.snapshot().raw_save == RAW_REPORT
        assert run.snapshot().analyses == {"primary": primary.snapshot, "post": capture}
        before = list(client.transport.sent)
        assert post.snapshot() == capture
        assert client.transport.sent == before
        methods = [method for method, _ in client.transport.sent]
        assert {
            method: methods.count(method)
            for method in (
                "tab.run_start",
                "tab.save_data",
                "tab.save_artifacts",
                "tab.post_analyze",
            )
        } == {
            "tab.run_start": 1,
            "tab.save_data": 1,
            "tab.save_artifacts": 2,
            "tab.post_analyze": 1,
        }
        assert (
            "tab.post_analyze",
            {"tab_id": "t", "updates": {}, "operation_id": 93, "run_operation_id": 71},
        ) in client.transport.sent
        assert [
            params
            for method, params in client.transport.sent
            if method == "tab.save_artifacts"
        ] == [
            {
                "tab_id": "t",
                "artifacts": [{"section": "analysis", "name": "results"}],
                "operation_id": 93,
            },
            {
                "tab_id": "t",
                "artifacts": [{"section": "post_analysis", "name": "results"}],
                "operation_id": 104,
            },
        ]
    finally:
        client.context.session.close()
