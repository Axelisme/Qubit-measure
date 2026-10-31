"""Recipe analysis parameters cross the native tracked owner boundary."""

from pathlib import Path

import pytest
from pydantic import BaseModel
from zcu_tools.analysis.failure import AnalysisFailed
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.plotting.plots import Plots

from tests.gui.app.measure._synthetic import (
    SampleAdapter,
    SampleAnalysis,
    SampleCfg,
    SampleParams,
    SampleResult,
)

from ._helpers import mcp_client
from ._recipe_analysis_adapter import RecipePicker


class Inputs(FormModel):
    pass


class Picked(BaseModel):
    flx_half: float
    flx_int: float
    flx_period: float


class Report(BaseModel):
    half: float
    written: int


@pytest.mark.parametrize("error_type", [AnalysisFailed, ValueError])
def test_analyze_errors_reach_recipe_author_with_original_message(
    fx, tmp_path: Path, request: pytest.FixtureRequest, error_type: type[Exception]
) -> None:
    message = "synthetic analyze could not produce a result"

    def analyze(
        req: AnalyzeRequest[RunRecord[SampleCfg, SampleResult], SampleParams],
        *,
        plots: Plots,
    ) -> SampleAnalysis:
        raise error_type(message)

    fx.registry.register(
        "synthetic/failing-analysis",
        Adapter(
            spec=SampleAdapter.spec,
            make_form=SampleAdapter.make_form,
            run=SampleAdapter.run,
            guide=None,
            analysis=Analysis(fn=analyze, options=lambda source, env: SampleParams()),
        ),
    )
    observed: list[str] = []

    @recipe("analyze-error", "Let the author handle any analyze failure")
    async def analyze_error(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic/failing-analysis")
        await tab.run()
        try:
            await tab.analyze("primary", result=Picked)
        except GuiRpcError as error:
            observed.append(str(error))
        else:
            raise AssertionError("Analyze failure must raise to its author")
        return Report(half=0.0, written=0)

    _, call = mcp_client(
        fx.service.port, tmp_path, request=request, recipes=(analyze_error,)
    )
    call("connect", {"port": fx.service.port})
    result = call("analyze-error", {})
    assert result["status"] == "completed", result
    assert observed == [message]


def test_updates_reach_owner_and_done_resumes_same_recipe_before_writeback_answer(
    fx, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    fx.registry.register("synthetic/recipe-picker", RecipePicker)
    fx.view.interactive_presentation.return_value = None
    context = fx.ctrl.context_control
    captured = context.read_context()
    assert captured is not None
    context.set_parameter(captured.target, "general.ext.recipe_half", 0.0)

    @recipe("pick", "Analyze with named native updates, then ask about writeback")
    async def pick(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic/recipe-picker")
        await tab.run()
        picked = await tab.analyze(
            "primary",
            result=Picked,
            interactive=True,
            updates={"tone": 4, "half": 1.25, "integer": -0.75},
        )
        assert picked.result is not None
        answer = await tab.ask_writeback()
        receipt = tab.accept_writeback(answer.items if answer.accepted else ())
        receipt.raise_for_error()
        return Report(half=picked.result.flx_half, written=len(receipt.items))

    _, call = mcp_client(fx.service.port, tmp_path, request=request, recipes=(pick,))
    call("connect", {"port": fx.service.port})
    started = call("pick", {})
    assert started["status"] == "interactive", started
    execution = started["execution"]
    pending = started["pending"]
    tab_id, operation = pending["tab"], pending["op"]
    read = call("tab_interact", {"tab": tab_id})
    assert read["state"]["flux_half"] == 1.25
    assert read["state"]["flux_int"] == -0.75
    assert call("status", {"execution": execution})["pending"] == pending
    rejected = call(
        "answer",
        {"execution": execution, "request_id": "not-a-question", "answer": {}},
    )
    assert rejected["accepted"] is False
    assert call("status", {"execution": execution})["pending"] == pending
    done = call("tab_interact", {"tab": tab_id, "payload": {"command": "done"}})
    assert done["execution"] == execution
    assert call("wait", {"op": operation})["status"] == "finished"
    status = call("status", {"execution": execution, "detail": "full"})
    analyses = status["details"][0]["analyses"]
    assert len(analyses) == 1
    assert analyses[0]["op"] == operation
    question = status["pending_question"]
    items = question["question"]["proposal"]["items"]
    assert items[0]["value"] == 1.25
    before_answer = context.read_context()
    assert before_answer is not None
    assert before_answer.lookup("general.ext.recipe_half").value == 0.0
    completed = call(
        "answer",
        {
            "execution": execution,
            "request_id": question["request_id"],
            "answer": {"accepted": True, "items": items},
        },
    )
    assert completed["status"] == "completed"
    assert completed["report"] == {"half": 1.25, "written": 1}
    after_answer = context.read_context()
    assert after_answer is not None
    assert after_answer.lookup("general.ext.recipe_half").value == 1.25
