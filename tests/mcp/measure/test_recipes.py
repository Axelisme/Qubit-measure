"""Recipe pipeline, handoff, parameters and reuse through the shipped tools."""

import base64
from dataclasses import asdict
from threading import Event, Thread, current_thread
from typing import Any

import pytest
from zcu_tools.mcp.core.reply import ToolReply
from zcu_tools.mcp.measure import tools_recipes

from ._recipe_actual_support import cfg_value
from ._recipe_stage_support import RAW_REPORT
from ._recipe_support import PNG, ExecutionGui, recipe_client
from ._recipe_sync_support import wait_for_event
from ._support import full_execution_reply, make_client


def test_recipe_observes_run_save_and_analysis_on_one_worker(tmp_path):
    gui = ExecutionGui()
    waiting_workers: set[Thread] = set()
    awaited_operations: set[object] = set()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.await":
            waiting_workers.add(current_thread())
            awaited_operations.add(params["operation_id"])
        return gui(method, params)

    with recipe_client(tmp_path, respond) as client:
        reply = full_execution_reply(
            client, client.call("pipeline", {"frequency_mhz": 6020.0})
        )
        assert reply.data["status"] == "awaiting_answer", reply.data
        assert reply.data["question_items"] == ("trigger_offset",)
        assert reply.data["written"] == ()
        assert reply.data["raw_save"] == asdict(RAW_REPORT)
        assert reply.data["analysis"]["result"]["summary"] == {"offset": 0.24}
        finished = client.call(
            "answer", {"recipe": reply.data["execution"], "decision": "skipped"}
        )
        assert finished.data["status"] == "finished"
        assert not finished.data["writeback"]["receipts"]
        assert awaited_operations == {71, 93}
        assert len(waiting_workers) == 1


def test_lookback_initial_wait_returns_while_the_same_execution_continues(
    tmp_path, monkeypatch
):
    gui = ExecutionGui()
    run_waiting = Event()
    release_run = Event()

    def respond(method, params):
        if method == "operation.await" and params["operation_id"] == 71:
            run_waiting.set()
            wait_for_event(release_run, "Test must release the original Run await")
        return gui(method, params)

    client = make_client(tmp_path, respond)
    # Exercise the tool's timeout branch without relying on worker scheduling speed.
    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0)
    try:
        returned = client.call("pipeline", {"frequency_mhz": 6020.0})
        wait_for_event(run_waiting, "Original Run must reach operation.await")
        assert returned.data["status"] == "running"
        initial = full_execution_reply(client, returned).data
        assert initial["status"] == "running"
        execution = initial["execution"]
        initial["actual"].clear()
        before = len(client.transport.sent)
        status = client.call("status", {"execution": execution, "detail": "full"})
        waiting = full_execution_reply(
            client, client.call("wait", {"execution": execution, "timeout": 0})
        )
        assert status["actual"]
        assert waiting.data["execution"] == execution
        assert waiting.data["status"] == "running"
        assert len(client.transport.sent) == before
        release_run.set()
        completed = full_execution_reply(
            # Released Run must reach the question; timeout only guards deadlock.
            client,
            client.call("wait", {"execution": execution, "timeout": 30}),
        )
        assert completed.data["status"] == "awaiting_answer", completed.data
        assert completed.data["run_op"] == status["run_op"]
        assert completed.data["raw_save"] == asdict(RAW_REPORT)
        assert completed.images[0].data == PNG
        finished = client.call("answer", {"recipe": execution, "decision": "skipped"})
        assert finished.data["status"] == "finished"
        assert not finished.data["writeback"]["receipts"]
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.analyze") == 1
    finally:
        release_run.set()
        client.context.session.close()


@pytest.fixture
def interactive_recipe(tmp_path, handoff_failure):
    gui = ExecutionGui()
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
            return {"reason": "timeout"}
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
    initial = client.call("pipeline", {"frequency_mhz": 6020.0})
    handoff = full_execution_reply(client, initial)
    assert handoff.data["status"] == "interactive"
    assert handoff.data["raw_save"] == asdict(RAW_REPORT)
    assert handoff.data["analysis"]["op"] == handoff.data["op"]
    assert initial.data["status"] == "interactive"
    assert initial.data["steps"]["run"]["status"] == "finished"
    assert initial.data["steps"]["raw_save"]["status"] == "saved"
    assert initial.data["steps"]["analysis"]["primary"]["status"] == "interactive"
    assert initial.data["artifacts"]["raw"] == asdict(RAW_REPORT)["artifacts"]["raw"]
    interaction = handoff.data["analysis"]["interaction"]
    assert interaction is not None
    if handoff_failure:
        assert handoff.is_error
        assert interaction["delivery_error"]
        assert not handoff.images
    else:
        assert not handoff.is_error
        assert handoff.images[0].data == PNG
    if handoff_failure not in ("query", "replaced"):
        assert interaction["state"] == {"offset": 0.24}
        assert interaction["commands"] == [{"name": "done"}]
    analysis_execution = handoff.data["analysis"]["execution"]
    execution = handoff.data["execution"]
    read = client.call("tab_interact", {"tab": "t"})
    assert read.data["prompt"] == "Confirm offset"
    assert read.images[0].data == PNG
    next_handoff = client.call(
        "tab_interact", {"tab": "t", "payload": {"command": "done"}}
    )
    assert next_handoff.data["status"] == "awaiting_answer"
    assert next_handoff.data["execution"] == execution
    assert next_handoff.data["question_items"] == ("trigger_offset",)
    assert not next_handoff.data["writeback"]["receipts"]
    wait_for_event(writeback_read, "Recipe must resume after interactive analysis")
    finished = full_execution_reply(
        # Released work must reach a handoff; timeout only guards deadlock.
        client,
        client.call("wait", {"execution": execution, "timeout": 30}),
    )
    assert finished.data["status"] == "awaiting_answer", finished.data
    assert finished.data["analysis"]["execution"] == analysis_execution
    assert read.data["execution"] == analysis_execution
    assert finished.data["writeback"]["has_draft"]
    assert finished.images[0].data == PNG
    answered = client.call("answer", {"recipe": execution, "decision": "skipped"})
    assert answered.data["status"] == "finished"
    assert answered.data["question_items"] is None
    assert not answered.data["writeback"]["receipts"]
    methods = [method for method, _ in client.transport.sent]
    assert methods.count("tab.run_start") == methods.count("tab.analyze") == 1


@pytest.mark.parametrize("replacement_source", [99, None])
@pytest.mark.parametrize("available", [False, True])
def test_lookback_rejects_a_post_run_snapshot_from_another_source(
    tmp_path, replacement_source, available
):
    gui = ExecutionGui()

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
            client, client.call("pipeline", {"frequency_mhz": 6020.0})
        )
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["error"]["reason"] == "result_superseded"
        assert reply.data["run_outcome"]["status"] == "finished"
        assert reply.data["result_state"] == {
            "available": available,
            "revision": 99,
            "source_operation_id": replacement_source,
        }
        assert reply.data["raw_save"] is None
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
    gui = ExecutionGui()
    with recipe_client(tmp_path, gui) as client:
        reply = full_execution_reply(
            client,
            client.call(
                "pipeline",
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
        assert data["status"] == "awaiting_answer", data
        assert not reply.is_error
        answered = client.call(
            "answer", {"recipe": data["execution"], "decision": "skipped"}
        )
        assert answered.data["status"] == "finished"
        assert not answered.data["writeback"]["receipts"]
        assert data["tab"] == "t"
        assert data["run_op"] != 71
        assert data["run_outcome"]["status"] == "finished"
        assert data["raw_save"] == asdict(RAW_REPORT)
        assert data["analysis"]["result"]["summary"] == {"offset": 0.24}
        assert data["analysis"]["save_report"]["run_id"] == "run-a"
        assert data["analysis"]["save_report"]["artifacts"]["analysis"]["results"] == {
            "status": "Saved",
            "members": {
                "analysis:primary": [
                    {
                        "format": "json",
                        "path": "/runs/run-a/analysis/results.json",
                        "status": "saved",
                        "error": None,
                    }
                ],
                "figure:trace": [
                    {
                        "format": "png",
                        "path": "/runs/run-a/analysis/trace.png",
                        "status": "saved",
                        "error": None,
                    }
                ],
            },
        }
        assert data["writeback"]["items"] == [
            {
                "id": "param-1",
                "kind": "parameter",
                "target_name": "trigger_offset",
                "selected": False,
                "proposed": 0.24,
                "current": 0.1,
            }
        ]
        assert reply.images[0].data == PNG
        assert data["elapsed_s"] >= 0
        for path, value in {
            "modules.readout.module.freq": 6020.0,
            "modules.readout.module.ro_length": expected[0],
            "modules.readout.module.trig_offset": expected[1],
            "rounds": 7,
        }.items():
            assert cfg_value(gui.publication, path) == value
            assert type(cfg_value(gui.publication, path)) is type(value)
        methods = [method for method, _ in client.transport.sent]
        assert methods.count("tab.run_start") == 1
        assert methods.count("tab.reset_cfg") == 1
        assert methods.count("tab.new") == int(not reuse)
        assert methods.count("tab.save_data") == 1
        assert methods.count("tab.analyze") == 1
        assert methods.index("device.snapshot") < methods.index("tab.run_start")
        assert methods.index("soc.info") < methods.index("tab.run_start")
        assert methods.index("tab.save_data") < methods.index("tab.analyze")
        assert methods.index("tab.save_artifacts") < methods.index(
            "tab.writeback_preview"
        )
        edits = [
            edit
            for method, params in client.transport.sent
            if method == "tab.edit_recipe_fields"
            for edit in params["edits"]
        ]
        by_path = {tuple(edit["path"]): edit["value"] for edit in edits}
        selections = [
            params
            for method, params in client.transport.sent
            if method == "tab.select_recipe_module"
        ]
        assert selections == [
            {"tab_id": "t", "path": ["modules", name], "action": "disable"}
            for name in ("reset", "init_pulse")
        ]
        for name in ("reset", "init_pulse"):
            assert cfg_value(gui.publication, f"modules.{name}") is None
        for key, value in zip(("ro_length", "trig_offset"), expected, strict=True):
            assert by_path["modules", "readout", key] == value
            assert type(by_path["modules", "readout", key]) is float


@pytest.mark.parametrize("reuse_tab_id", [None, "kept"])
def test_lookback_missing_frequency_does_not_run_a_blind_default(
    tmp_path, reuse_tab_id
):
    tab = reuse_tab_id or "new"
    gui = ExecutionGui()

    def respond(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "tab.parameter_read":
            return {
                "values": {path: None for path in params["paths"]},
                "modules": {reference: None for reference in params["module_refs"]},
            }
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
        if method in (
            "tab.get_cfg",
            "tab.reset_cfg",
            "tab.read_recipe_field",
            "tab.select_recipe_module",
        ):
            return gui(method, params)
        raise AssertionError(f"Missing frequency must not start work: {method}")

    with recipe_client(tmp_path, respond) as client:
        arguments = {} if reuse_tab_id is None else {"reuse_tab_id": reuse_tab_id}
        reply = client.call("pipeline", arguments)
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
        assert methods.count("tab.reset_cfg") == 1


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
    gui = ExecutionGui()
    gui.md["$resonator.freq"] = 6500.0
    with recipe_client(tmp_path, gui) as client:
        reply = client.call("pipeline", arguments)
        assert reply.is_error
        assert reply.data["status"] == "failed"
        assert reply.data["tab"] is None
        assert not gui.ran
        assert not any(method.startswith("tab.") for method, _ in client.transport.sent)


@pytest.mark.parametrize("failure", ["wrong_experiment", "busy", "missing", "stale"])
def test_lookback_reuse_errors_do_not_create_replacement_or_retry(tmp_path, failure):
    gui = ExecutionGui()
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
        reply = client.call("pipeline", {"reuse_tab_id": "t", "frequency_mhz": 6000.0})
        assert reply.is_error
        assert reply.data["tab"] == "t"
        methods = [method for method, _ in client.transport.sent]
        assert "tab.new" not in methods
        assert "tab.run_start" not in methods
        assert methods.count("tab.reset_cfg") == int(failure == "stale")
    finally:
        client.context.session.close()
