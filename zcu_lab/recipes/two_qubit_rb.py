"""Run one reference/CZ-interleaved two-qubit RB scan and save its data."""

from pydantic import Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from zcu_lab.v2.two_qubit.rb.core import RBAnalysis

from .two_qubit_rb_helpers import RBEstimateReport, RBRecipeInputs, rb_estimate_fields


class RBReport(RBEstimateReport):
    """Paired RB estimate and independent diagnostics for its two fitted arms."""

    reference_fit_quality: FitQuality | None = Field(
        default=None,
        description="Reference arm fit diagnostics, or null without a fit.",
    )
    interleaved_fit_quality: FitQuality | None = Field(
        default=None,
        description="CZ-interleaved arm fit diagnostics, or null without a fit.",
    )


@recipe(
    "two_qubit_rb",
    description="Compare classified joint ground return for reference and CZ-interleaved two-qubit RB, report the CZ error estimate, and save without writeback.",
)
async def two_qubit_rb(session: RecipeSession, inputs: RBRecipeInputs) -> RBReport:
    """Fit the two paired RB arms and report their CZ error estimate.

    See [Interpreting the result](../v2/two_qubit/rb/guide.md#interpreting-the-result).
    Set depths (random Cliffords) and seeds (paired sequence identities).
    Omission keeps the Reset value; explicit null goes to the Form.
    Use overrides for pulse, readout and optional reset edits.
    Save after a completed Run even if analysis fails. Do not ask or write back.
    """
    tab = session.open_tab(
        "two_qubit/rb",
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
        analysis = await tab.analyze("primary", result=RBAnalysis)
        if analysis.result is None:
            return RBReport(reason=analysis.reason, analysis_failure=analysis.reason)

        result = analysis.result
        reference, interleaved = (
            result.fit_quality if result.fit_quality is not None else (None, None)
        )
        return RBReport(
            **rb_estimate_fields(result.estimate),
            reason=result.reason,
            analysis_failure=result.reason,
            reference_fit_quality=reference,
            interleaved_fit_quality=interleaved,
        )
    finally:
        tab.save()


DEFINITION = two_qubit_rb
