"""Generator execution contracts through injected definitions and recording GUI."""

from dataclasses import replace
from threading import Event

import pytest
from zcu_tools.mcp.measure.recipe import (
    MissingParameter,
    RecipeAnalysis,
    RecipeGenerator,
    RecipeInputSchema,
    RecipeNeedsParameters,
    RecipeRun,
    RecipeSession,
)
from zcu_tools.mcp.measure.recipe_execution import RecipeExecutions
from zcu_tools.mcp.measure.session import GuiRpcError

from ._recipe_analysis_support import AnalysisRecipeGui
from ._recipe_execution_support import ControlledRunGui, definition, registry
from ._recipe_support import LookbackGui, recipe_client


def test_generator_run_analysis_and_snapshot_capture_do_not_query_gui(tmp_path):
    finally_ran = Event()
    deliveries: list[str] = []

    def sample(session: RecipeSession, *, gain: float) -> RecipeGenerator:
        try:
            assert type(gain) is float
            tab = session.open_tab("lookback")
            tab.set("rounds", int(gain))
            run, status = yield tab.run()
            assert isinstance(run, RecipeRun)
            deliveries.append(status)
            run.save_raw()
            primary, status = yield run.analyze("primary")
            assert isinstance(primary, RecipeAnalysis)
            deliveries.append(status)
            post, status = yield run.analyze("post")
            assert isinstance(post, RecipeAnalysis)
            deliveries.append(status)
        finally:
            finally_ran.set()

    schema: RecipeInputSchema = {
        "type": "object",
        "properties": {"gain": {"type": "number"}},
        "required": ["gain"],
        "additionalProperties": False,
    }
    gui = AnalysisRecipeGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample, schema)) as executions,
    ):
        execution = executions.start(client.context, "sample", {"gain": 4})
        reply = execution.wait(5)
        assert reply.data["status"] == "finished" and not reply.is_error
        assert deliveries == ["completed", "completed", "completed"]
        assert finally_ran.is_set()
        snapshot = execution.snapshot()
        assert snapshot.actual is not None
        assert snapshot.raw_save.path == "/actual/raw.h5"
        assert snapshot.analysis is not None and snapshot.post_analysis is not None
        assert snapshot.analysis.stage == "primary"
        assert snapshot.post_analysis.stage == "post"
        assert snapshot.writeback is not None and snapshot.post_writeback is not None
        assert executions.get(snapshot.execution) is execution
        calls = list(client.transport.sent)
        assert snapshot.analysis.params is not None
        snapshot.analysis.params["threshold"] = -1
        current = execution.snapshot()
        assert current.analysis is not None
        assert current.analysis.params == {"threshold": 0.5}
        assert executions.snapshots()[0] == execution.snapshot()
        execution.wait(0)
        execution.close()
        execution.join()
        assert client.transport.sent == calls
        assert finally_ran.is_set()


def test_two_interactive_stages_resume_in_background_without_tool_wait(tmp_path):
    reached_post = Event()
    resumed = Event()

    def sample(session: RecipeSession) -> RecipeGenerator:
        run, _ = yield session.open_tab("lookback").run()
        assert isinstance(run, RecipeRun)
        primary, _ = yield run.analyze("primary")
        assert isinstance(primary, RecipeAnalysis)
        reached_post.set()
        post, _ = yield run.analyze("post")
        assert isinstance(post, RecipeAnalysis)
        resumed.set()

    gui = AnalysisRecipeGui(interactive=True)
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        first = execution.wait(5)
        assert first.data["status"] == "interactive"
        assert first.data["analysis_stage"] == "primary"
        assert first.images
        assert not reached_post.is_set()
        gui.done["primary"].set()
        assert reached_post.wait(2)
        second = execution.wait(5)
        assert second.data["status"] == "interactive"
        assert second.data["analysis_stage"] == "post"
        assert not resumed.is_set()
        gui.done["post"].set()
        assert resumed.wait(2)
        final = execution.wait(5)
        assert final.data["status"] == "finished"
        assert (
            final.data["analysis"]["writeback"]["items"][0]["target_name"]
            == "frequency"
        )
        assert (
            final.data["post_analysis"]["writeback"]["items"][0]["target_name"]
            == "linewidth"
        )


@pytest.mark.parametrize("answer", ["accepted", "skipped"])
def test_question_captures_items_and_answer_only_delivers_decision(tmp_path, answer):
    delivered: list[object] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        run, _ = yield session.open_tab("lookback").run()
        assert isinstance(run, RecipeRun)
        _, _ = yield run.analyze("primary")
        decision, status = yield run.propose_writeback(["frequency"])
        delivered.extend((decision, status))

    gui = AnalysisRecipeGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        question = execution.wait(5)
        assert question.data["status"] == "awaiting_answer"
        assert question.data["question_items"] == ("frequency",)
        assert (
            question.data["question_preview"]["primary"]["items"][0]["proposed"] == 2.0
        )
        with pytest.raises(GuiRpcError) as busy:
            executions.start(client.context, "sample", {})
        assert busy.value.reason == "recipe_busy"
        assert question.data["execution"] in str(busy.value)
        reply = execution.answer(answer)
        assert reply.data["status"] == "finished"
        assert delivered == [answer, "completed"]
        assert reply.data["question_items"] is None
        assert not any(
            method == "tab.writeback_accept" for method, _ in client.transport.sent
        )
        with pytest.raises(GuiRpcError) as duplicate:
            execution.answer(answer)
        assert duplicate.value.reason == "not_awaiting_answer"


def test_question_cancel_delivers_none_and_clears_one_intent(tmp_path):
    delivered: list[object] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        run, _ = yield session.open_tab("lookback").run()
        assert isinstance(run, RecipeRun)
        _, _ = yield run.analyze("primary")
        decision, status = yield run.propose_writeback()
        delivered.extend((decision, status))

    gui = AnalysisRecipeGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        assert execution.wait(5).data["status"] == "awaiting_answer"
        execution.cancel()
        assert execution.wait(5).data["status"] == "cancelled"
        assert delivered == [None, "cancelled"]
        assert not execution.snapshot().cancel_requested


@pytest.mark.parametrize("control", ["cancel", "finish_early", "gui_cancel"])
def test_native_run_stop_delivers_partial_handle_and_correct_status(tmp_path, control):
    delivered: list[str] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        run, status = yield session.open_tab("lookback").run()
        assert isinstance(run, RecipeRun)
        delivered.append(status)
        if status == "finished_early":
            run.save_raw()

    gui = ControlledRunGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        assert gui.awaited.wait(2)
        with pytest.raises(GuiRpcError) as busy:
            executions.start(client.context, "sample", {})
        assert busy.value.reason == "recipe_busy"
        if control == "gui_cancel":
            gui.outcome = "cancelled"
            gui.settled.set()
        else:
            reply = (
                execution.cancel() if control == "cancel" else execution.finish_early()
            )
            assert reply.data["gui_cancel"]["status"] == "requested"
        reply = execution.wait(5)
        expected = "finished_early" if control == "finish_early" else "cancelled"
        assert delivered == [expected]
        assert reply.data["status"] == (
            "finished" if control == "finish_early" else "cancelled"
        )
        assert reply.data["run_outcome"]["status"] == "cancelled"
        assert gui.raw_saved == (control == "finish_early")


def test_interactive_cancel_delivers_none_without_another_analysis_worker(tmp_path):
    delivered: list[object] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        run, _ = yield session.open_tab("lookback").run()
        assert isinstance(run, RecipeRun)
        analysis, status = yield run.analyze("primary")
        delivered.extend((analysis, status))

    gui = AnalysisRecipeGui(interactive=True)
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        assert execution.wait(5).data["status"] == "interactive"
        with pytest.raises(GuiRpcError) as error:
            execution.finish_early()
        assert error.value.reason == "not_running"
        execution.cancel()
        assert execution.wait(5).data["status"] == "cancelled"
        assert delivered == [None, "cancelled"]


def test_between_yield_cancel_suppresses_once_but_keeps_fast_steps(tmp_path):
    between = Event()
    continue_recipe = Event()
    delivered: list[object] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        tab = session.open_tab("lookback")
        run, _ = yield tab.run()
        assert isinstance(run, RecipeRun)
        between.set()
        assert continue_recipe.wait(2)
        tab.set("rounds", 9)
        analysis, status = yield run.analyze("primary")
        delivered.extend((analysis, status))
        analysis, status = yield run.analyze("primary")
        assert isinstance(analysis, RecipeAnalysis)
        delivered.append(status)

    gui = AnalysisRecipeGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        assert between.wait(2)
        try:
            execution.cancel()
        finally:
            continue_recipe.set()
        reply = execution.wait(5)
        assert reply.data["status"] == "finished"
        assert delivered == [None, "cancelled", "completed"]
        assert (
            len(
                [
                    method
                    for method, _ in client.transport.sent
                    if method == "tab.analyze"
                ]
            )
            == 1
        )
        publication = client.context.send_gui_rpc("tab.get_cfg", {"tab_id": "t"})
        assert publication["tree"]["children"]["rounds"]["input"]["resolved"] == 9


@pytest.mark.parametrize("caught", [False, True])
def test_native_failure_is_thrown_at_yield_and_finally_runs(tmp_path, caught):
    finally_ran = Event()
    failures: list[str | None] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            try:
                _, _ = yield session.open_tab("lookback").run()
            except GuiRpcError as error:
                failures.append(error.reason)
                if not caught:
                    raise
        finally:
            finally_ran.set()

    gui = ControlledRunGui()
    gui.outcome = "failed"
    gui.settled.set()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        reply = execution.wait(5)
        assert reply.data["status"] == ("finished" if caught else "failed")
        assert failures == ["run_failed"]
        assert reply.data["run_outcome"]["status"] == "failed"
        assert finally_ran.is_set()
        assert reply.is_error == (not caught)


@pytest.mark.parametrize("interactive", [False, True])
def test_close_serializes_finally_and_never_resumes_a_closed_generator(
    tmp_path, interactive
):
    finally_ran = Event()
    resumed = Event()

    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            run, _ = yield session.open_tab("lookback").run()
            assert isinstance(run, RecipeRun)
            _, _ = yield run.analyze("primary")
            resumed.set()
        finally:
            finally_ran.set()

    gui = AnalysisRecipeGui(interactive=True) if interactive else ControlledRunGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        if interactive:
            assert execution.wait(5).data["status"] == "interactive"
        else:
            assert isinstance(gui, ControlledRunGui)
            assert gui.awaited.wait(2)
        executions.stop_admission()
        executions.join()
        assert finally_ran.is_set()
        assert not resumed.is_set()
        assert execution.snapshot().status == "cancelled"
        execution.close()
        execution.join()
        with pytest.raises(GuiRpcError) as closed:
            executions.start(client.context, "sample", {})
        assert closed.value.reason == "session_closed"


def test_needs_parameters_and_invalid_keywords_fail_without_starting_run(tmp_path):
    finally_ran = Event()

    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            raise RecipeNeedsParameters((MissingParameter("frequency", "Supply MHz"),))
            yield session.open_tab("lookback").run()
        finally:
            finally_ran.set()

    with (
        recipe_client(tmp_path, LookbackGui()) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        before = list(client.transport.sent)
        rejected = executions.start(client.context, "sample", {"extra": 1}).wait(5)
        assert rejected.is_error and rejected.data["status"] == "failed"
        assert rejected.data["error"]["phase"] == "preparing"
        assert "extra" in rejected.data["error"]["message"]
        assert not finally_ran.is_set()
        assert client.transport.sent == before
        execution = executions.start(client.context, "sample", {})
        reply = execution.wait(5)
        assert reply.data["status"] == "needs_parameters"
        assert reply.data["missing"][0]["parameter"] == "frequency"
        assert finally_ran.is_set()
        assert not any(method == "tab.run_start" for method, _ in client.transport.sent)


def test_registry_validates_definitions_and_rejects_unknown_identity(tmp_path):
    def sample(session: RecipeSession) -> RecipeGenerator:
        yield session.open_tab("lookback").run()

    valid = definition(sample)
    for invalid in (
        replace(valid, name=""),
        replace(valid, adapter_name=" "),
        replace(
            valid,
            input_schema={
                "type": "object",
                "properties": {},
                "required": ["undeclared"],
            },
        ),
    ):
        with pytest.raises(ValueError, match="Recipe requires|Invalid recipe schema"):
            RecipeExecutions(Event(), recipes=(invalid,))
    with pytest.raises(ValueError, match="Duplicate"):
        RecipeExecutions(Event(), recipes=(valid, valid))
    with (
        recipe_client(tmp_path, LookbackGui()) as client,
        registry(client.context, valid) as executions,
    ):
        with pytest.raises(GuiRpcError) as name:
            executions.start(client.context, "unknown", {})
        assert name.value.reason == "unknown_recipe"
        with pytest.raises(GuiRpcError) as identity:
            executions.get("missing")
        assert identity.value.reason == "unknown_execution"
        first = executions.start(client.context, "sample", {})
        first.wait(5)
        calls = list(client.transport.sent)
        with pytest.raises(GuiRpcError) as binding:
            executions.start(replace(client.context), "sample", {})
        assert binding.value.reason == "wrong_binding"
        assert client.transport.sent == calls
        for timeout in (-1, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="timeout"):
                first.wait(timeout)
