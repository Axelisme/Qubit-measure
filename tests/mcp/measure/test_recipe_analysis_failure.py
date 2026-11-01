"""Unhandled analyze errors fail recipes and retain the original message."""

from pathlib import Path

import pytest

from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from ._script_recipe_support import Fit, Inputs, Report, SyntheticGui, start
from ._support import make_client


@pytest.mark.parametrize(
    "diagnostic",
    ["fit did not converge", "ValueError()", "ValueError('no result')"],
)
def test_unhandled_analysis_failure_ends_recipe(tmp_path: Path, diagnostic: str) -> None:
    gui = SyntheticGui()
    observed: list[tuple[str, object, str | None]] = []

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "operation.await" and params["operation_id"] == 100:
            return {
                "reason": "completed",
                "status": "failed",
                "error": diagnostic,
            }
        return gui.respond(method, params)

    @recipe("synthetic", "Let the author decide after a failed fit")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        fit = await tab.analyze("primary", result=Fit)
        observed.append((fit.status, fit.result, fit.reason))
        tab.save()
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        snapshot = execution.snapshot()
        assert snapshot.status == "failed"
        assert observed == []
        assert snapshot.error is not None
        assert snapshot.error.message == diagnostic
    finally:
        client.context.session.close()
