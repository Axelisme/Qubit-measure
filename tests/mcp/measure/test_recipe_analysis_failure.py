"""Expected analysis failures are values; programming errors still end recipes."""

from pathlib import Path

import pytest
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from ._script_recipe_support import Fit, Inputs, Report, SyntheticGui, start
from ._support import make_client


@pytest.mark.parametrize("classified", [True, False])
def test_analysis_failure_leaves_the_author_in_control(
    tmp_path: Path, classified: bool
) -> None:
    gui = SyntheticGui()
    observed: list[tuple[str, object, str | None]] = []

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.await" and params["operation_id"] == 100:
            return {
                "reason": "completed",
                "status": "failed",
                "error": "fit did not converge",
                "failure_kind": "analysis" if classified else None,
            }
        return gui.respond(method, params)

    @recipe("synthetic", "Let the author decide after a failed fit")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        fit = await tab.analyze("primary", result=Fit)
        observed.append((fit.status, fit.result, fit.reason))
        # A failed fit need not discard the acquisition or fail this recipe.
        tab.save()
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        snapshot = execution.snapshot()
        if classified:
            assert snapshot.status == "completed", snapshot.error
            assert observed == [("failed", None, "fit did not converge")]
            assert snapshot.outcome is not None
            assert len(snapshot.outcome.runs[0].saves) == 1
        else:
            assert snapshot.status == "failed"
            assert observed == []
            assert snapshot.error is not None
            assert snapshot.error.message == "fit did not converge"
    finally:
        client.context.session.close()
