"""Named recipe analysis updates stay on the same tracked operation."""

from pathlib import Path

from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from ._script_recipe_support import Fit, Inputs, Report, SyntheticGui, start
from ._support import make_client


def test_post_analysis_updates_share_the_started_operation(tmp_path: Path) -> None:
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.get_post_analyze_result":
            return {"summary": {"estimate": 7.25}, "invalid": []}
        return gui.respond(method, params)

    @recipe("synthetic", "Send named post-analysis updates on its tracked operation")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        await tab.analyze("primary", result=Fit)
        fit = await tab.analyze(
            "post", result=Fit, updates={"threshold": 7.25, "enabled": True}
        )
        assert fit.result is not None and fit.result.estimate == 7.25
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        starts = [
            (method, params)
            for method, params in client.transport.sent
            if method in {"tab.analyze", "tab.post_analyze"}
        ]
        assert starts == [
            ("tab.analyze", {"tab_id": "tab-1", "updates": {}, "run_operation_id": 1}),
            (
                "tab.post_analyze",
                {
                    "tab_id": "tab-1",
                    "updates": {"threshold": 7.25, "enabled": True},
                    "run_operation_id": 1,
                    "operation_id": 100,
                },
            ),
        ]
    finally:
        client.context.session.close()
