"""Run paired two-qubit RB with a herald readout before every sequence."""

from pydantic import Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from zcu_lab.v2.heralded.rb.core import RBHeraldedAnalysis

from .two_qubit_rb_helpers import RBEstimateReport, RBRecipeInputs, rb_estimate_fields


class RBHeraldedReport(RBEstimateReport):
    """Four-class joint estimate with gg and ee fit diagnostics in the report."""

    gg_reference_fit_quality: FitQuality | None = Field(
        default=None, description="gg herald, reference arm fit diagnostics, or null."
    )
    gg_interleaved_fit_quality: FitQuality | None = Field(
        default=None,
        description="gg herald, CZ-interleaved arm fit diagnostics, or null.",
    )
    ee_reference_fit_quality: FitQuality | None = Field(
        default=None, description="ee herald, reference arm fit diagnostics, or null."
    )
    ee_interleaved_fit_quality: FitQuality | None = Field(
        default=None,
        description="ee herald, CZ-interleaved arm fit diagnostics, or null.",
    )


@recipe(
    "heralded_rb",
    description="Fit gg, ge, eg and ee return jointly for paired two-qubit RB. Report the shared CZ error estimate and gg/ee diagnostics, then save without writeback.",
)
async def heralded_rb(
    session: RecipeSession, inputs: RBRecipeInputs
) -> RBHeraldedReport:
    """Fit four herald classes jointly and report the gg and ee diagnostics.

    See [Joint decay and uncertainty](../v2/heralded/rb/guide.md#joint-decay-and-uncertainty).
    Set depths (random Cliffords) and seeds (paired sequence identities).
    Omission keeps the Reset value; explicit null goes to the Form.
    Every shot includes a herald read. Optional reset overrides accept ordinary
    reset only. Save after a completed Run even if analysis fails; no writeback.
    """
    tab = session.open_tab(
        "heralded/rb",
        target=inputs.target.aliases() if inputs.target is not None else None,
        reuse=inputs.reuse_tab_id,
    )
    tab.set("depth", inputs.depths)
    tab.set("seed", inputs.seeds)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=RBHeraldedAnalysis)
        if analysis.result is None:
            return RBHeraldedReport(
                reason=analysis.reason, analysis_failure=analysis.reason
            )

        result = analysis.result
        if result.fit_quality is None:
            gg_reference = gg_interleaved = ee_reference = ee_interleaved = None
        else:
            # Analysis orders gg, ge, eg, ee, each reference then interleaved.
            (
                gg_reference,
                gg_interleaved,
                _ge_reference,
                _ge_interleaved,
                _eg_reference,
                _eg_interleaved,
                ee_reference,
                ee_interleaved,
            ) = result.fit_quality
        return RBHeraldedReport(
            **rb_estimate_fields(result.estimate),
            reason=result.reason,
            analysis_failure=result.reason,
            gg_reference_fit_quality=gg_reference,
            gg_interleaved_fit_quality=gg_interleaved,
            ee_reference_fit_quality=ee_reference,
            ee_interleaved_fit_quality=ee_interleaved,
        )
    finally:
        tab.save()


DEFINITION = heralded_rb
