"""Sweep relative phase and pulse duration, then review the Qubit2 phase update."""

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

from zcu_lab.v2.two_qubit.constructive_phase.core import ConstructivePhaseAnalysis

from .two_qubit_helpers import TwoQubitTarget, apply_selected_writeback


class ConstructivePhaseInputs(RecipeInputs):
    """Four-role target and phase/duration sweep inputs. Overrides apply last."""

    target: TwoQubitTarget | None = Field(
        default=None,
        description="Position bindings for this Reset. Omitted or null positions keep the current aliases.",
    )
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    relative_phase_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP,
        description="Start and stop of the Qubit2 minus Qubit1 phase sweep, in deg.",
    )
    relative_phase_points: int | None | Keep = Field(
        KEEP, description="Number of relative phase samples, mapped to Form Expts."
    )
    duration_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP, description="Start and stop of the common pulse duration sweep, in us."
    )
    duration_points: int | None | Keep = Field(
        KEEP, description="Number of common duration samples, mapped to Form Expts."
    )
    reps: int | None | Keep = Field(
        KEEP, description="Repetitions per point per round."
    )
    rounds: int | None | Keep = Field(KEEP, description="Acquisition round count.")


class ConstructivePhaseReport(RecipeReport):
    """Owner's constructive phase estimate and interference fit quality."""

    relative_phase_deg: float | None = Field(
        default=None,
        description="Maximum-rate Qubit2 minus Qubit1 phase, in deg; null if unavailable.",
    )
    reason: str | None = Field(
        default=None,
        description="Owner reason when no relative phase can be identified.",
    )


@recipe(
    "two_qubit_constructive_phase",
    description="Scan Qubit2 relative phase and common duration, review Qubit2 phase writeback and save.",
)
async def two_qubit_constructive_phase(
    session: RecipeSession, inputs: ConstructivePhaseInputs
) -> ConstructivePhaseReport:
    """Find the Qubit2 minus Qubit1 phase of maximum fitted rate.

    See [Read the figures](../v2/two_qubit/constructive_phase/guide.md#read-the-figures).
    Set relative_phase_range in deg and duration_range in us. Both point
    counts set Form Expts. Omission keeps the Reset value. Null ranges clear
    both bounds; other nulls pass to the Form. Use overrides for pulses,
    readout and reset. Ask about Primary writeback only when the fit identifies
    a relative phase. Save after a completed Run, even if phase is unresolved.
    """
    tab = session.open_tab(
        "two_qubit/constructive_phase",
        target=inputs.target.aliases() if inputs.target is not None else None,
        reuse=inputs.reuse_tab_id,
    )
    if inputs.relative_phase_range is None:
        tab.set("relative_phase.start", None)
        tab.set("relative_phase.stop", None)
    elif not isinstance(inputs.relative_phase_range, Keep):
        start, stop = inputs.relative_phase_range
        tab.set("relative_phase.start", start)
        tab.set("relative_phase.stop", stop)
    tab.set("relative_phase.n", inputs.relative_phase_points)
    if inputs.duration_range is None:
        tab.set("duration.start", None)
        tab.set("duration.stop", None)
    elif not isinstance(inputs.duration_range, Keep):
        start, stop = inputs.duration_range
        tab.set("duration.start", start)
        tab.set("duration.stop", stop)
    tab.set("duration.n", inputs.duration_points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ConstructivePhaseAnalysis)
        if analysis.result is None:
            return ConstructivePhaseReport(analysis_failure=analysis.reason)

        fit = analysis.result
        phase = report_number(fit.relative_phase_deg)
        applied = False
        if phase is not None:
            decision = await tab.ask_writeback(stage="primary")
            applied = apply_selected_writeback(tab, decision)

        return ConstructivePhaseReport(
            relative_phase_deg=phase,
            reason=fit.reason,
            fit_quality=fit.interference_quality,
            analysis_failure=fit.reason,
            writeback_applied=applied,
        )
    finally:
        tab.save()


DEFINITION = two_qubit_constructive_phase
