"""Typed helper deliveries through the public recipe driver, without lab recipes."""

from threading import Event

import pytest
from zcu_tools.mcp.measure import recipe
from zcu_tools.mcp.measure.recipe import RecipeGenerator, RecipeSession

from ._recipe_analysis_support import AnalysisRecipeGui
from ._recipe_execution_support import (
    ControlledRunGui,
    completed,
    definition,
    handoff,
    registry,
    wait_for_event,
)
from ._recipe_support import recipe_client


def test_helpers_deliver_named_results_and_preserve_answer_handoff(tmp_path):
    delivered: list[
        recipe.RunStepResult | recipe.AnalysisStepResult | recipe.WritebackResult
    ] = []

    def sample(session: RecipeSession) -> RecipeGenerator:
        result = yield from session.open_tab("lookback").run_step()
        delivered.append(result)
        if result.run is None:
            return
        analysis = yield from result.run.analyze_step("primary")
        delivered.append(analysis)
        decision = yield from recipe.ask(result.run.propose_writeback())
        delivered.append(decision)

    with (
        recipe_client(tmp_path, AnalysisRecipeGui()) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        assert handoff(execution).data["status"] == "awaiting_answer"
        assert execution.answer("accepted").data["status"] == "finished"
        assert isinstance(delivered[0], recipe.RunStepResult)
        assert isinstance(delivered[1], recipe.AnalysisStepResult)
        assert isinstance(delivered[2], recipe.WritebackResult)
        assert [result.status for result in delivered] == ["completed"] * 3
        assert delivered[2].decision == "accepted"
        assert not any(
            method == "tab.writeback_accept" for method, _ in client.transport.sent
        )


@pytest.mark.parametrize("control", ["cancel", "finish_early"])
def test_run_helper_exposes_stop_status_and_allows_author_cleanup(tmp_path, control):
    delivered: list[recipe.RunStepResult] = []
    cleaned = Event()
    ready = Event()

    class ReadyRunGui(ControlledRunGui):
        def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
            reply = super().__call__(method, params)
            if method == "operation.await":
                ready.set()
            return reply

    def sample(session: RecipeSession) -> RecipeGenerator:
        try:
            result = yield from session.open_tab("lookback").run_step()
            delivered.append(result)
        finally:
            cleaned.set()
            ready.set()

    gui = ReadyRunGui()
    with (
        recipe_client(tmp_path, gui) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        execution = executions.start(client.context, "sample", {})
        # The worker either reaches native await or exits through author cleanup.
        wait_for_event(ready)
        if gui.awaited.is_set():
            if control == "cancel":
                execution.cancel()
            else:
                execution.finish_early()
        reply = completed(execution)
        assert reply.data["status"] == (
            "cancelled" if control == "cancel" else "finished"
        )
        assert len(delivered) == 1
        assert delivered[0].status == (
            "cancelled" if control == "cancel" else "finished_early"
        )
        assert delivered[0].run is not None
        assert cleaned.is_set()


def test_direct_yield_operation_is_rejected_with_helper_guidance(tmp_path):
    def sample(session: RecipeSession) -> RecipeGenerator:
        yield session.open_tab("lookback").run()

    with (
        recipe_client(tmp_path, AnalysisRecipeGui()) as client,
        registry(client.context, definition(sample)) as executions,
    ):
        reply = completed(executions.start(client.context, "sample", {}))
        assert reply.data["status"] == "failed"
        assert "yield from" in reply.data["error"]["message"]
        assert "run_step" in reply.data["error"]["message"]
