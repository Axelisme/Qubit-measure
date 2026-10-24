"""Named recipe analysis updates stay on the same tracked operation."""

from pathlib import Path
from typing import Literal

import pytest
from zcu_tools.format_version import YamlValue
from zcu_tools.mcp.measure.recipe_api import KEEP, Keep, RecipeSession, recipe

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


@pytest.mark.parametrize("stage", ["primary", "post"])
@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        (
            {
                "retained": KEEP,
                "also_retained": Keep(),
                "threshold": None,
                "enabled": False,
                "skip": 0,
            },
            {"threshold": None, "enabled": False, "skip": 0},
        ),
        ({"retained": KEEP, "also_retained": Keep()}, {}),
        (None, {}),
    ],
    ids=["mixed", "all-keep", "omitted"],
)
def test_analysis_updates_omit_keep_and_preserve_null(
    tmp_path: Path,
    stage: Literal["primary", "post"],
    updates: dict[str, YamlValue | Keep] | None,
    expected: dict[str, YamlValue],
) -> None:
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.get_post_analyze_result":
            return {"summary": {"estimate": 3.5}, "invalid": []}
        return gui.respond(method, params)

    @recipe("synthetic", "Forward only explicit analysis edits to the owner")
    async def synthetic(session: RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        await tab.run()
        if stage == "post":
            await tab.analyze("primary", result=Fit)
        await tab.analyze(stage, result=Fit, updates=updates)
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        starts = [
            params
            for method, params in client.transport.sent
            if method in {"tab.analyze", "tab.post_analyze"}
        ]
        assert starts[-1]["updates"] == expected
    finally:
        client.context.session.close()
