"""Injected generator tools, recipe-owned done, answers and actual write receipts."""

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from zcu_tools.mcp.measure.assembly import build_measure_tools
from zcu_tools.mcp.measure.execution_reply import SummaryEstimate
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeRun, RecipeSession
from zcu_tools.mcp.measure.session import GuiRpcError

from ._analyze_support import call_stdio, stdio_data
from ._recipe_analysis_support import AnalysisRecipeGui
from ._recipe_execution_support import ControlledRunGui, definition
from ._support import make_client


def _sample(
    session: RecipeSession, *, post: bool = False, write: bool = False
) -> RecipeGenerator:
    tab = session.open_tab("lookback")
    run, status = yield tab.run()
    if status == "cancelled":
        return
    assert isinstance(run, RecipeRun)
    run.save_raw()
    _, status = yield run.analyze("primary")
    if status == "cancelled":
        return
    if post:
        _, status = yield run.analyze("post")
        if status == "cancelled":
            return
    decision, status = yield run.propose_writeback()
    if status != "cancelled" and decision == "accepted" and write:
        tab.accept()


SAMPLE = replace(
    definition(_sample),
    input_schema={
        "type": "object",
        "properties": {"post": {"type": "boolean"}, "write": {"type": "boolean"}},
        "additionalProperties": False,
    },
    summary_estimates=(SummaryEstimate("frequency", "frequency", unit="MHz"),),
)


class WritebackGui(AnalysisRecipeGui):
    """Expose current drafts independently from completed proposal operation IDs."""

    def __init__(self, *, interactive: bool = False) -> None:
        super().__init__(interactive=interactive)
        self.post_started = False
        self.writes: list[dict[str, object]] = []

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "adapter.guide":
            return {"guide": {"adapter": params["adapter_name"]}}
        if method == "tab.post_analyze":
            self.post_started = True
        if "operation_id" not in params:
            if method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
                return {
                    "summary": {}
                    if method == "tab.get_analyze_result" or self.post_started
                    else None
                }
            if method == "tab.writeback_preview":
                stage = "primary" if params["subtab_id"] == "analysis" else "post"
                return dict(deepcopy(self.writebacks[stage]))
        if method == "tab.writeback_write":
            self.writes.append(deepcopy(params))
            requested = params["write"]
            assert isinstance(requested, list)
            return {
                "written": [
                    {
                        "id": item["id"],
                        "kind": "md",
                        "target": "frequency",
                        "before": {"value": 1.0},
                        "after": {"value": 2.0},
                    }
                    for item in requested
                ]
            }
        return super().__call__(method, params)


@pytest.mark.parametrize(
    "decision,write", [("skipped", True), ("accepted", False), ("accepted", True)]
)
def test_answer_reports_actual_writes_not_decision_intent(tmp_path, decision, write):
    gui = WritebackGui()
    client = make_client(tmp_path, gui, recipes=(SAMPLE,))
    with closing(client.context.session):
        question = client.call("sample", {"write": write})
        key = question.data["execution"]
        assert question.data["status"] == "awaiting_answer"
        assert question.data["question_items"] == ("frequency",)
        assert question.data["writeback"]["receipts"] == ()
        calls = list(client.transport.sent)
        summary = client.call("status", {"execution": key})
        full = client.call("status", {"execution": key, "detail": "full"})
        waited = client.call("wait", {"execution": key, "timeout": 0})
        assert (
            summary["question_items"] == waited.data["question_items"] == ("frequency",)
        )
        assert full["question_preview"]["primary"]["items"][0]["id"] == "primary-draft"
        assert client.transport.sent == calls
        # Answer reads current panes, not the proposal operation's old draft IDs.
        gui.writebacks["primary"]["items"][0]["id"] = "current-draft"
        result = client.call("answer", {"recipe": key, "decision": decision})
        assert result.data["execution"] == key
        assert result.data["status"] == "finished" and not result.is_error
        assert result.data["question_items"] is None
        receipts = result.data["writeback"]["receipts"]
        if decision == "accepted" and write:
            assert len(gui.writes) == len(receipts) == 1
            assert gui.writes[0]["write"] == [{"id": "current-draft"}]
            assert receipts[0]["completed"][0]["written"][0]["after"] == {"value": 2.0}
            receipts[0]["completed"].clear()
            assert client.call("status", {"execution": key})["writeback"]["receipts"][
                0
            ]["completed"]
        else:
            assert gui.writes == [] and receipts == ()
        with pytest.raises(GuiRpcError) as error:
            client.call("answer", {"recipe": key, "decision": decision})
        assert error.value.reason == "not_awaiting_answer"


def test_primary_and_post_done_return_recipe_continuation_then_question(tmp_path):
    gui = WritebackGui(interactive=True)
    client = make_client(tmp_path, gui, recipes=(SAMPLE,))
    with closing(client.context.session):
        primary = client.call("sample", {"post": True})
        key = primary.data["execution"]
        assert primary.data["status"] == "interactive"
        assert primary.data["interaction"]["state"] == {"stage": "primary"}
        post = client.call("tab_interact", {"tab": "t", "payload": {"command": "done"}})
        assert post.data["execution"] == key and post.data["status"] == "interactive"
        assert post.data["analysis"]["stage"] == "post"
        reads = [
            method for method, _ in client.transport.sent if method != "operation.await"
        ]
        status = client.call("status", {"execution": key})
        waited = client.call("wait", {"execution": key, "timeout": 0})
        full = client.call("status", {"execution": key, "detail": "full"})
        assert [
            method for method, _ in client.transport.sent if method != "operation.await"
        ] == reads
        for summary in (post.data, status, waited.data):
            assert summary["interaction"]["state"] == {"stage": "post"}
            assert summary["interaction"]["commands"] == [
                {"name": "select-post"},
                {"name": "done"},
            ]
            assert summary["interaction"]["info"] == {"label": "post-picker"}
        assert full["analysis"]["interaction"]["state"] == {"stage": "primary"}
        assert full["analysis"]["interaction"]["commands"] == [
            {"name": "select-primary"},
            {"name": "done"},
        ]
        assert full["analysis"]["interaction"]["info"] == {"label": "primary-picker"}
        assert full["post_analysis"]["interaction"]["state"] == {"stage": "post"}
        question = client.call(
            "tab_interact", {"tab": "t", "payload": {"command": "done"}}
        )
        assert question.data["execution"] == key
        assert question.data["status"] == "awaiting_answer"
        assert question.data["question_items"] == ("frequency", "linewidth")
        assert len(question.data["previews"]["primary"]) == 1
        assert len(question.data["previews"]["post"]) == 1
        result = client.call("answer", {"recipe": key, "decision": "skipped"})
        assert result.data["status"] == "finished"
        assert gui.writes == []


@pytest.mark.parametrize(
    ("failed_stage", "post_is_error"), [("primary", False), ("post", True)]
)
def test_delivery_error_is_scoped_to_the_current_interactive_stage(
    tmp_path, failed_stage, post_is_error
):
    gui = WritebackGui(interactive=True)

    def respond(method, params):
        reply = gui(method, params)
        if method == "tab.interact" and gui.current_stage == failed_stage:
            reply["figure"] = {"png_b64": "invalid-png"}
        return reply

    client = make_client(tmp_path, respond, recipes=(SAMPLE,))
    with closing(client.context.session):
        primary = client.call("sample", {"post": True})
        key = primary.data["execution"]
        assert primary.data["status"] == "interactive"
        assert primary.is_error is (failed_stage == "primary")
        # GUI-local done does not replace Primary interaction through an MCP call.
        gui.done["primary"].set()
        assert gui.awaited["post"].wait(2)
        post = client.call("wait", {"execution": key, "timeout": 5})
        assert post.data["status"] == "interactive"
        assert post.data["analysis"]["stage"] == "post"
        assert post.is_error is post_is_error
        full = client.call("status", {"execution": key, "detail": "full"})
        pane = "analysis" if failed_stage == "primary" else "post_analysis"
        assert "delivery_error" in full[pane]["interaction"]
        gui.done["post"].set()
        question = client.call("wait", {"execution": key, "timeout": 5})
        assert question.data["status"] == "awaiting_answer" and not question.is_error
        client.call("answer", {"recipe": key, "decision": "skipped"})


def test_interactive_delivery_error_is_preserved_without_failing_native_operation(
    tmp_path,
):
    gui = WritebackGui(interactive=True)
    client = make_client(tmp_path, gui, recipes=(SAMPLE,))
    client.transport.replies["tab.interact"] = {
        "ok": True,
        "result": {
            "operation_id": 93,
            "plugin": "generic-test",
            "state": {},
            "info": {},
            "commands": [{"name": "done"}],
            "preview_active": False,
            "figure": {"png_b64": "invalid-png"},
        },
    }
    with closing(client.context.session):
        reply = client.call("sample", {})
        key = reply.data["execution"]
        assert reply.data["status"] == "interactive" and reply.is_error
        assert client.call("wait", {"execution": key, "timeout": 0}).is_error
        assert (
            "delivery_error"
            in client.call("status", {"execution": key, "detail": "full"})["analysis"][
                "interaction"
            ]
        )
        del client.transport.replies["tab.interact"]
        question = client.call(
            "tab_interact", {"tab": "t", "payload": {"command": "done"}}
        )
        assert question.data["status"] == "awaiting_answer" and not question.is_error
        client.call("answer", {"recipe": key, "decision": "skipped"})


@pytest.mark.parametrize("control", ["cancel", "finish_early"])
def test_op_control_resolves_original_run_and_preserves_native_outcome(
    tmp_path, monkeypatch, control
):
    from zcu_tools.mcp.measure import tools_recipes

    monkeypatch.setattr(tools_recipes, "INITIAL_WAIT_SECONDS", 0.01)
    controlled = ControlledRunGui()
    gui = WritebackGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if (
            method in ("operation.await", "operation.cancel")
            and params["operation_id"] == 71
        ):
            return controlled(method, params)
        return gui(method, params)

    client = make_client(tmp_path, respond, recipes=(SAMPLE,))
    with closing(client.context.session):
        pending = client.call("sample", {})
        assert controlled.awaited.wait(2)
        key = pending.data["execution"]
        run_op = client.call("status", {"execution": key})["run_op"]
        reply = client.call(control, {"op": run_op})
        assert reply.data["execution"] == key
        assert controlled.cancel_count == 1
        result = client.call("wait", {"execution": key, "timeout": 2})
        assert result.data["status"] == (
            "cancelled" if control == "cancel" else "awaiting_answer"
        )
        assert not result.is_error
        if control == "finish_early":
            assert result.data["artifacts"]["raw"]["data"]["status"] == "saved"
            client.call("answer", {"recipe": key, "decision": "skipped"})
        else:
            assert not gui.raw_saved


@pytest.mark.parametrize("route", ["execution", "op"])
def test_analysis_cancel_routes_to_the_recipe_not_a_separate_execution(tmp_path, route):
    gui = WritebackGui(interactive=True)
    client = make_client(tmp_path, gui, recipes=(SAMPLE,))
    with closing(client.context.session):
        pending = client.call("sample", {})
        key = pending.data["execution"]
        target = (
            {"execution": key} if route == "execution" else {"op": pending.data["op"]}
        )
        control = client.call("cancel", target)
        assert control.data["execution"] == key
        result = client.call("wait", {"execution": key, "timeout": 2})
        assert result.data["status"] == "cancelled" and not result.is_error
        assert gui.writes == []


def test_question_blocks_new_recipe_and_nonrun_finish_early(tmp_path):
    gui = WritebackGui()
    client = make_client(tmp_path, gui, recipes=(SAMPLE,))
    with closing(client.context.session):
        question = client.call("sample", {})
        key = question.data["execution"]
        before = list(client.transport.sent)
        with pytest.raises(GuiRpcError, match=key) as error:
            client.call("sample", {})
        assert error.value.reason == "recipe_busy"
        with pytest.raises(GuiRpcError) as error:
            client.call("finish_early", {"execution": key})
        assert error.value.reason == "not_running"
        assert client.transport.sent == before
        client.call("cancel", {"execution": key})
        result = client.call("wait", {"execution": key, "timeout": 2})
        assert result.data["status"] == "cancelled"


def test_session_close_drains_generator_finally_before_removing_previews(tmp_path):
    observed: list[bool] = []
    finally_ran = Event()
    preview: list[str] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            run, _ = yield session.open_tab("lookback").run()
            assert isinstance(run, RecipeRun)
            _, _ = yield run.analyze("primary")
            _, _ = yield run.propose_writeback()
        finally:
            observed.append(bool(preview) and Path(preview[0]).is_file())
            finally_ran.set()

    client = make_client(tmp_path, WritebackGui(), recipes=(definition(sample),))
    question = client.call("sample", {})
    preview.extend(question.data["previews"]["primary"])
    client.context.session.close()
    assert finally_ran.is_set() and observed == [True]
    assert all(not Path(path).exists() for path in preview)
    with pytest.raises(GuiRpcError) as error:
        client.call("sample", {})
    assert error.value.reason == "session_closed"


def test_injected_guide_and_estimates_share_the_owning_registry(tmp_path):
    gui = WritebackGui()
    declared = replace(SAMPLE, name="custom", adapter_name="declared-adapter")
    client = make_client(tmp_path, gui, recipes=(declared,))
    with closing(client.context.session):
        question = client.call("custom", {})
        guide = client.call("recipe_guide", {"recipe": "custom"})
        assert guide["adapter"] == guide["guide"]["adapter"] == "declared-adapter"
        # Analysis-only has no recipe identity; it uses supplied scalar declarations.
        client.transport.replies["tab.analyze"] = {
            "ok": True,
            "result": {
                "operation_id": 117,
                "interactive": False,
                "params": {},
                "invalidated_on_success": [],
            },
        }
        client.transport.replies["operation.await"] = {
            "ok": True,
            "result": {"reason": "completed", "status": "finished"},
        }
        client.transport.replies["tab.get_analyze_result"] = {
            "ok": True,
            "result": {
                "summary": {
                    "frequency": 5.0,
                    "fit_quality": {"fit": {"r2": 1.0, "invalid": []}},
                },
                "params": {},
                "invalid": [],
                "operation_state": {"analysis_state": {"figure_names": []}},
            },
        }
        client.transport.replies["tab.writeback_preview"] = {
            "ok": True,
            "result": {"has_draft": False, "items": [], "destination_context": {}},
        }
        reply = client.call("tab_analyze", {"tab": "t"})
        assert reply.data["recipe"] is None
        assert (
            reply.data["analysis"]["primary"]["estimates"]["frequency"]["value"] == 5.0
        )
        client.call(
            "answer", {"recipe": question.data["execution"], "decision": "skipped"}
        )
        with pytest.raises(ValueError, match="unknown recipe"):
            client.call("recipe_guide", {"recipe": "lookback"})


def test_tool_assembly_refuses_inconsistent_or_colliding_injection(tmp_path):
    client = make_client(tmp_path, recipes=(SAMPLE,))
    with closing(client.context.session):
        with pytest.raises(ValueError, match="owning session"):
            build_measure_tools(client.context, recipes=())
        assert client.transport.sent == []
    with pytest.raises(RuntimeError, match="duplicate MCP tool"):
        make_client(tmp_path, recipes=(replace(SAMPLE, name="answer"),))


@pytest.mark.parametrize(
    "arguments,match",
    [
        ({"frequency_mhz": True}, "Invalid recipe argument frequency_mhz"),
        ({"frequency_mhz": float("nan")}, "finite"),
        ({"rounds": 1.0}, "rounds"),
        ({"readout_ref": ""}, "Invalid recipe argument readout_ref"),
        ({"unknown": 1}, "Additional properties"),
    ],
)
def test_production_lookback_rejects_invalid_inputs_before_gui_access(
    tmp_path, arguments, match
):
    client = make_client(tmp_path, WritebackGui())
    with closing(client.context.session):
        rejected = client.call("lookback", arguments)
        assert rejected.is_error and rejected.data["status"] == "failed"
        assert match in rejected.data["error"]["message"]
        full = client.call(
            "status", {"execution": rejected.data["execution"], "detail": "full"}
        )
        assert full["error"]["phase"] == "preparing"
        assert full["status"] == "failed"
        assert client.transport.sent == []


def test_production_lookback_preserves_sources_and_only_writes_after_answer(tmp_path):
    gui = WritebackGui()
    client = make_client(tmp_path, gui)
    with closing(client.context.session):
        question = client.call(
            "lookback",
            {
                "frequency_mhz": 5100,
                "readout_length_us": 4,
                "trigger_offset_us": 0.25,
                "rounds": 7,
            },
        )
        assert question.data["status"] == "awaiting_answer"
        actual = question.data["actual"]["parameters"]
        for name in (
            "frequency_mhz",
            "readout_length_us",
            "trigger_offset_us",
            "rounds",
        ):
            assert actual[name]["source"] == name
        assert actual["frequency_mhz"]["value"] == 5100.0
        assert gui.raw_saved and gui.writes == []
        result = client.call(
            "answer", {"recipe": question.data["execution"], "decision": "accepted"}
        )
        assert result.data["status"] == "finished" and not result.is_error
        assert (
            result.data["writeback"]["receipts"][0]["completed"][0]["stage"]
            == "primary"
        )


def test_answer_write_failure_retains_confirmed_primary_prefix(tmp_path):
    gui = WritebackGui()
    client = make_client(tmp_path, gui, recipes=(SAMPLE,))
    with closing(client.context.session):
        question = client.call("sample", {"post": True, "write": True})

        def write_then_fail(params):
            if params["subtab_id"] == "analysis":
                return {"ok": True, "result": gui("tab.writeback_write", params)}
            return {
                "ok": False,
                "error": {
                    "code": "internal_error",
                    "message": "post apply interrupted",
                },
            }

        client.transport.replies["tab.writeback_write"] = write_then_fail
        result = client.call(
            "answer", {"recipe": question.data["execution"], "decision": "accepted"}
        )
        assert result.data["status"] == "failed" and result.is_error
        receipt = result.data["writeback"]["receipts"][0]
        assert receipt["failed_stage"] == "post"
        assert receipt["failed_stage_may_have_partial_writes"] is True
        assert receipt["completed"][0]["stage"] == "primary"
        assert receipt["completed"][0]["written"][0]["id"] == "primary-draft"
        assert receipt["not_started"] == []
        assert len(gui.writes) == 1
        full = client.call(
            "status", {"execution": question.data["execution"], "detail": "full"}
        )
        assert full["written"][0] == receipt


def test_standalone_apply_writeback_does_not_answer_or_claim_recipe_writes(tmp_path):
    gui = WritebackGui()
    client = make_client(tmp_path, gui, recipes=(SAMPLE,))
    with closing(client.context.session):
        question = client.call("sample", {})
        key = question.data["execution"]
        receipt = client.call("apply_writeback", {"tab": "t"})
        assert receipt["status"] == "finished" and len(gui.writes) == 1
        summary = client.call("status", {"execution": key})
        assert summary["status"] == "awaiting_answer"
        assert summary["writeback"]["receipts"] == ()
        result = client.call("answer", {"recipe": key, "decision": "skipped"})
        assert result.data["status"] == "finished" and len(gui.writes) == 1


@pytest.mark.parametrize(
    "arguments,match",
    [
        ({"decision": "accepted"}, "recipe must"),
        ({"recipe": ""}, "recipe must"),
        ({"recipe": "sample", "decision": True}, "decision must"),
        ({"recipe": "sample", "decision": "invalid"}, "decision must"),
        (
            {"recipe": "sample", "decision": "accepted", "extra": True},
            "Unexpected answer fields",
        ),
    ],
)
def test_answer_rejects_contract_inputs_without_gui_access(tmp_path, arguments, match):
    client = make_client(tmp_path, recipes=(SAMPLE,))
    with closing(client.context.session):
        with pytest.raises(ValueError, match=match):
            client.call("answer", arguments)
        assert client.transport.sent == []


def test_question_and_answer_share_json_stdio_projection(tmp_path, monkeypatch):
    gui = WritebackGui()
    client = make_client(tmp_path, gui)
    with closing(client.context.session):
        question = stdio_data(
            call_stdio(monkeypatch, client, "lookback", {"frequency_mhz": 5100})
        )
        assert question["status"] == "awaiting_answer"
        assert question["question_items"] == ["frequency"]
        assert question["writeback"]["receipts"] == []
        result = stdio_data(
            call_stdio(
                monkeypatch,
                client,
                "answer",
                {
                    "recipe": question["execution"],
                    "decision": "accepted",
                },
            )
        )
        assert result["execution"] == question["execution"]
        assert result["status"] == "finished"
        assert result["writeback"]["receipts"][0]["completed"][0]["written"][0][
            "after"
        ] == {"value": 2.0}
