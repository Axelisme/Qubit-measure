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
        tab = session.prepare_tab(
            "synthetic", inputs, target_alias="qubit", values={("gain",): inputs.gain}
        )
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
                "library": None,
                "inline": None,
                "overrides": None,
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
        ] == ([[]] if value is None else [[{"path": ["gain"], "value": value}]])
    finally:
        client.context.session.close()


@pytest.mark.parametrize("reuse", [None, "tab-1"])
def test_prepare_tab_orders_sources_values_and_overrides(
    tmp_path: Path, reuse: str | None
) -> None:
    class Inputs(api.RecipeInputs):
        count: int = 9

    @api.recipe("synthetic", "Prepare sources and one complete experiment batch")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        session.prepare_tab(
            "synthetic",
            inputs,
            target_alias="qubit",
            values={("count",): inputs.count, ("retained",): None},
        )
        return Report(count=inputs.count)

    client = make_client(tmp_path, SyntheticGui().respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context,
            "synthetic",
            {
                "target": "Q2",
                "reuse_tab_id": reuse,
                "library": [
                    {"path": ["tones", 3, "pulse"], "key": "first"},
                    {"path": ["tones", 1, "pulse"], "key": "second"},
                    {"path": ["retained_selector"], "key": None},
                ],
                "inline": [
                    {"path": ["tones", 3, "pulse"]},
                    {"path": ["tones", 3, "pulse", "child"]},
                ],
                "overrides": [
                    {"path": ["count"], "value": 17},
                    {"path": ["count"], "value": None},
                ],
            },
        )
        assert isinstance(execution, ScriptRecipeExecution)
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        effects = [
            (method, params)
            for method, params in client.transport.sent
            if method.startswith("tab.") and method != "tab.snapshot"
        ]
        assert effects == (
            [("tab.new", {"adapter_name": "synthetic"})] if reuse is None else []
        ) + [
            ("tab.reset_cfg", {"tab_id": "tab-1", "aliases": {"qubit": "Q2"}}),
            (
                "tab.select_recipe_module",
                {
                    "tab_id": "tab-1",
                    "path": ["tones", 3, "pulse"],
                    "action": "use_library",
                    "key": "first",
                },
            ),
            (
                "tab.select_recipe_module",
                {
                    "tab_id": "tab-1",
                    "path": ["tones", 1, "pulse"],
                    "action": "use_library",
                    "key": "second",
                },
            ),
            (
                "tab.select_recipe_module",
                {
                    "tab_id": "tab-1",
                    "path": ["tones", 3, "pulse"],
                    "action": "use_inline",
                },
            ),
            (
                "tab.select_recipe_module",
                {
                    "tab_id": "tab-1",
                    "path": ["tones", 3, "pulse", "child"],
                    "action": "use_inline",
                },
            ),
            (
                "tab.edit_recipe_fields",
                {
                    "tab_id": "tab-1",
                    "edits": [
                        {"path": ["count"], "value": 9},
                        {"path": ["count"], "value": 17},
                    ],
                    "require_valid": True,
                },
            ),
        ]
    finally:
        client.context.session.close()


@pytest.mark.parametrize("split", [True, False])
def test_custom_steps_can_precede_validation_or_replace_defaults(
    tmp_path: Path, split: bool
) -> None:
    class Inputs(api.RecipeInputs):
        pass

    gui = SyntheticGui()
    custom_installed = False

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        nonlocal custom_installed
        if method == "tab.edit_recipe_fields":
            if not params.get("require_valid"):
                custom_installed = True
            elif not custom_installed:
                return {"errors": [{"path": ["pulse"], "message": "Required"}]}
        return gui.respond(method, params)

    @api.recipe("synthetic", "Install an authored pulse before checking the cfg")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        if split:
            tab = session.open_recipe_tab("synthetic", inputs, target_alias="mux")
            tab.select_sources(inputs)
        else:
            tab = session.open_tab("synthetic")
            tab.select(("pulse",), ref="custom")
        # This experiment step is not a hook or part of a template method.
        tab.set(("pulse", "gain"), 0.25)
        if split:
            tab.apply_inputs(inputs, {("retained",): None})
        else:
            tab.apply([], require_valid=True)
        return Report(count=1)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context, "synthetic", {}
        )
        assert isinstance(execution, ScriptRecipeExecution)
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        edits = [
            params
            for method, params in client.transport.sent
            if method == "tab.edit_recipe_fields"
        ]
        assert edits == [
            {"tab_id": "tab-1", "edits": [{"path": ["pulse", "gain"], "value": 0.25}]},
            {"tab_id": "tab-1", "edits": [], "require_valid": True},
        ]
        reset = [p for m, p in client.transport.sent if m == "tab.reset_cfg"]
        assert reset == [{"tab_id": "tab-1"}]
    finally:
        client.context.session.close()


@pytest.mark.parametrize("require_valid", [True, False])
def test_apply_inputs_preserves_owner_validation_failures(
    tmp_path: Path, require_valid: bool
) -> None:
    class Inputs(api.RecipeInputs):
        pass

    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.edit_recipe_fields":
            return {
                "errors": [{"path": ["count"], "message": "Required"}],
                "changed": not require_valid,
            }
        return gui.respond(method, params)

    @api.recipe("synthetic", "Keep the owner's admission and draft failure semantics")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        session.prepare_tab(
            "synthetic",
            inputs,
            target_alias="qubit",
            values={("other",): 1},
            require_valid=require_valid,
        )
        return Report(count=1)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context, "synthetic", {}
        )
        assert isinstance(execution, ScriptRecipeExecution)
        execution.join()
        snapshot = execution.snapshot()
        assert snapshot.status == ("needs_parameters" if require_valid else "failed")
        if require_valid:
            assert snapshot.parameter_errors[0].path == ("count",)
            assert snapshot.parameter_errors[0].message == "Required"
        else:
            assert snapshot.error is not None
            assert snapshot.error.message == "Owner retained an invalid cfg draft"
        assert gui.runs == 0
    finally:
        client.context.session.close()


@pytest.mark.parametrize("rejected", [False, True])
def test_apply_inputs_admits_values_named_edits_and_overrides_together(
    tmp_path: Path, rejected: bool
) -> None:
    class Inputs(api.RecipeInputs):
        pass

    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.edit_recipe_fields" and rejected:
            return {
                "changed": False,
                "errors": [{"path": ["pulse", "length"], "message": "Required"}],
            }
        return gui.respond(method, params)

    @api.recipe("synthetic", "Admit derived edits and final overrides as one candidate")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_recipe_tab("synthetic", inputs, target_alias="qubit")
        tab.apply_inputs(
            inputs,
            {("count",): 3, ("retained",): None},
            edits=(
                api.FieldEdit(path=("count",), value=7, source="$calibration.count"),
                api.FieldEdit(path=("retained",), value=None, source="linewidth"),
            ),
        )
        return Report(count=1)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context,
            "synthetic",
            {"overrides": [{"path": ["count"], "value": 11}]},
        )
        execution.join()
        snapshot = execution.snapshot()
        assert snapshot.status == ("needs_parameters" if rejected else "completed"), (
            snapshot
        )
        assert [
            p for m, p in client.transport.sent if m == "tab.edit_recipe_fields"
        ] == [
            {
                "tab_id": "tab-1",
                "edits": [
                    {"path": ["count"], "value": 3},
                    {"path": ["count"], "value": 7},
                    {"path": ["count"], "value": 11},
                ],
                "require_valid": True,
            }
        ]
        if rejected:
            assert [(e.path, e.message) for e in snapshot.parameter_errors] == [
                (("pulse", "length"), "Required")
            ]
        assert gui.runs == 0
    finally:
        client.context.session.close()


def test_apply_inputs_retains_named_edit_source_validation(tmp_path: Path) -> None:
    @api.recipe("synthetic", "Keep the supplied edit source for native admission")
    async def synthetic(session: api.RecipeSession, inputs: api.RecipeInputs) -> Report:
        tab = session.open_recipe_tab("synthetic", inputs, target_alias="qubit")
        tab.apply_inputs(
            inputs,
            {},
            edits=(api.FieldEdit(path=("count",), value=7, source=" "),),
        )
        return Report(count=1)

    client = make_client(tmp_path, SyntheticGui().respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context, "synthetic", {}
        )
        execution.join()
        snapshot = execution.snapshot()
        assert snapshot.status == "failed"
        assert snapshot.error is not None
        assert snapshot.error.message == "Edit source must be nonblank"
        assert not [
            p for m, p in client.transport.sent if m == "tab.edit_recipe_fields"
        ]
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
        tab = session.prepare_tab("synthetic", inputs, target_alias="qubit")
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
