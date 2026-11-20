"""Scan the common CZ gain multiplier and review the first-return calibration."""

from typing import Annotated

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    Overrides,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    ReuseTabId,
    recipe,
    report_number,
)

from zcu_lab.v2.two_qubit.cz_amplitude.core import CZAmplitudeAnalysis

from .two_qubit_helpers import TwoQubitTarget, apply_selected_writeback


class CZAmplitudeInputs(RecipeInputs):
    """Four-role target and scan inputs. Overrides apply after named edits."""

    target: TwoQubitTarget | None = Field(
        default=None,
        description="Role bindings for this Reset. Omitted or null roles keep the current aliases.",
    )
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    scale_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP, description="Start and stop of the dimensionless common gain multiplier."
    )
    scale_points: int | None | Keep = Field(
        KEEP, description="Number of scale samples, mapped to Form Expts."
    )
    reps: int | None | Keep = Field(
        KEEP, description="Repetitions per point per round."
    )
    rounds: int | None | Keep = Field(KEEP, description="Acquisition round count.")


class CZAmplitudeReport(RecipeReport):
    """Owner's first-return fit, with null for unavailable or nonfinite values."""

    scale: float | None = Field(
        default=None,
        description="Dimensionless first-return scale within the measured span.",
    )
    period: float | None = Field(
        default=None,
        description="Fitted period in gain-scale units, even without a valid return.",
    )
    error: float | None = Field(
        default=None, description="One-sigma period uncertainty in gain-scale units."
    )
    offset: float | None = Field(default=None, description="Fitted probability offset.")
    contrast: float | None = Field(
        default=None, description="Signed cosine coefficient."
    )
    reason: str | None = Field(
        default=None,
        description="Owner reason when the fit cannot propose a return scale.",
    )


@recipe(
    "two_qubit_cz_amplitude",
    description="Scan common CZ gain, fit the first joint return, review pulse writeback and save.",
)
async def two_qubit_cz_amplitude(
    session: RecipeSession, inputs: CZAmplitudeInputs
) -> CZAmplitudeReport:
    """Sweep both CZ gains by one multiplier and fit the joint return.

    See [Fit the first complete return](../v2/two_qubit/cz_amplitude/guide.md#fit-the-first-complete-return).
    Set dimensionless scale_range and scale_points (scale.n → Expts).
    Omission keeps the Reset value. Null endpoints clear the sweep bounds;
    other nulls pass to the Form. Use overrides for pulses, readout and reset.
    Ask before writing the selected Primary candidates when a return exists.
    Save after a completed Run, including when analysis cannot find a return.
    """
    tab = session.open_tab(
        "two_qubit/cz_amplitude",
        target=inputs.target.aliases() if inputs.target is not None else None,
        reuse=inputs.reuse_tab_id,
    )
    if inputs.scale_range is None:
        tab.set("scale.start", None)
        tab.set("scale.stop", None)
    elif not isinstance(inputs.scale_range, Keep):
        start, stop = inputs.scale_range
        tab.set("scale.start", start)
        tab.set("scale.stop", stop)
    tab.set("scale.n", inputs.scale_points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=CZAmplitudeAnalysis)
        if analysis.result is None:
            return CZAmplitudeReport(analysis_failure=analysis.reason)

        fit = analysis.result.fit
        applied = False
        if fit.scale is not None:
            decision = await tab.ask_writeback(stage="primary")
            applied = apply_selected_writeback(tab, decision)

        return CZAmplitudeReport(
            scale=report_number(fit.scale),
            period=report_number(fit.period),
            error=report_number(fit.error),
            offset=report_number(fit.offset),
            contrast=report_number(fit.contrast),
            reason=fit.reason,
            fit_quality=fit.fit_quality,
            analysis_failure=fit.reason,
            writeback_applied=applied,
        )
    finally:
        tab.save()


DEFINITION = two_qubit_cz_amplitude
