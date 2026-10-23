"""Optional recipe defaults compose with authored steps through the wire seam."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.analysis.fitting.quality import FitQuality, QualityIssue
from zcu_tools.mcp.measure import recipe_api as api
from zcu_tools.mcp.measure.script_recipe_execution import ScriptRecipeExecution

from ._script_recipe_support import Report, SyntheticGui
from ._support import make_client


@dataclass(frozen=True)
class OwnerSummary:
    """Synthetic analysis owner's estimate and diagnostics, not a recipe view."""

    estimate: float
    quality: FitQuality


@pytest.mark.parametrize(
    ("arguments", "error_kind", "value"),
    [
        pytest.param({"typo": 1}, "extra_forbidden", None, id="unknown-field"),
        pytest.param({"gain": float("nan")}, "finite_number", 0.0, id="nan"),
        pytest.param({"gain": float("inf")}, "finite_number", -2.5, id="inf"),
        pytest.param({"gain": -float("inf")}, "finite_number", 1.25, id="neg-inf"),
    ],
)
def test_recipe_input_admission_rejects_invalid_values_before_measurement(
    tmp_path: Path, arguments: dict[str, object], error_kind: str, value: float | None
) -> None:
    class Inputs(api.RecipeInputs):
        gain: float | None = None

    gui = SyntheticGui()
    received: list[float | None] = []

    @api.recipe("synthetic", "Validate experiment inputs before acquisition")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        received.append(inputs.gain)
        tab = session.open_tab(
            "synthetic", target={"qubit": inputs.target}, reuse=inputs.reuse_tab_id
        )
        tab.set("gain", inputs.gain)
        tab.check_ready()
        await tab.run()
        return Report(count=1)

    client = make_client(tmp_path, gui.respond, recipes=(synthetic,))
    try:
        rejected = client.context.session.recipes.start(
            client.context, "synthetic", arguments
        )
        rejected.join()
        snapshot = rejected.snapshot()
        assert snapshot.status == "failed", snapshot
        assert snapshot.error is not None
        assert error_kind in snapshot.error.message
        assert snapshot.outcome is not None
        assert snapshot.outcome.status == "failed"
        assert received == []
        assert client.transport.sent == []
        assert gui.runs == 0

        accepted = client.context.session.recipes.start(
            client.context,
            "synthetic",
            {
                "gain": value,
                "target": None,
                "reuse_tab_id": None,
                "overrides": [],
            },
        )
        accepted.join()
        assert accepted.snapshot().status == "completed", accepted.snapshot()
        assert received == [value]
        assert gui.runs == 1
        assert [
            p["edits"]
            for m, p in client.transport.sent
            if m == "tab.edit_recipe_fields"
        ] == [[{"path": ["gain"], "value": value}]]
    finally:
        client.context.session.close()


def test_owner_analysis_diagnostics_flow_into_recipe_report(tmp_path: Path) -> None:
    class Inputs(api.RecipeInputs):
        pass

    class OwnerReport(api.RecipeReport):
        estimate: float | None

    gui = SyntheticGui()
    quality = FitQuality(
        -0.5, None, {"frequency": None}, (QualityIssue("r2", "warning"),)
    )

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.get_analyze_result":
            return {
                "summary": {
                    "estimate": 1.234567890123,
                    "quality": quality.to_summary_dict(),
                },
                "invalid": [],
            }
        return gui.respond(method, params)

    @api.recipe("synthetic", "Report the analysis owner's named result and issues")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> OwnerReport:
        tab = session.open_tab("synthetic")
        await tab.run()
        fit = await tab.analyze("primary", result=OwnerSummary)
        assert fit.result is not None
        assert isinstance(fit.result.quality, api.FitQuality)
        assert isinstance(fit.result.quality.invalid[0], api.QualityIssue)
        return OwnerReport(
            estimate=api.report_number(fit.result.estimate),
            fit_quality=fit.result.quality,
            diagnostics=fit.result.quality.invalid,
        )

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context, "synthetic", {}
        )
        assert isinstance(execution, ScriptRecipeExecution)
        execution.join()
        snapshot = execution.snapshot()
        assert snapshot.status == "completed", snapshot
        assert snapshot.outcome is not None
        assert snapshot.outcome.report is not None
        assert snapshot.outcome.report.model_dump(mode="json") == {
            "estimate": 1.234567890123,
            "writeback_applied": False,
            "analysis_failure": None,
            "fit_quality": {
                "r2": -0.5,
                "normalized_residual_rms": None,
                "relative_parameter_errors": {"frequency": None},
                "invalid": [{"path": "r2", "reason": "warning"}],
            },
            "diagnostics": [{"path": "r2", "reason": "warning"}],
        }
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (-float("inf"), None),
        (0.0, 0.0),
        (-0.0, -0.0),
        (-1.234567890123, -1.234567890123),
        (1e-30, 1e-30),
    ],
)
def test_report_number_retains_finite_precision(
    value: float | None, expected: float | None
) -> None:
    assert api.report_number(value) == expected
