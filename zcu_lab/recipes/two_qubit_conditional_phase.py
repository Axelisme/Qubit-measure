"""Scan common CZ frequency and review the conditional-phase crossing."""

from typing import Annotated, Literal

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

from zcu_lab.v2.two_qubit.conditional_phase.core import ConditionalPhaseAnalysis

from .two_qubit_helpers import TwoQubitTarget, apply_selected_writeback


class ConditionalPhaseInputs(RecipeInputs):
    """Scan inputs shared by both variants. Overrides apply after named edits."""

    target: TwoQubitTarget | None = Field(
        default=None,
        description="Position bindings for this Reset. Omitted or null positions keep the current aliases.",
    )
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    ramsey_role: Literal["qubit1", "qubit2"] | None | Keep = Field(
        KEEP,
        description="Form Ramsey target: the qubit that measures phase; the other is the spectator.",
    )
    gate_count: list[Annotated[int, Field(strict=True, ge=1)]] | None | Keep = Field(
        KEEP,
        description="Form CZ gates per shot (count): consecutive positive CZ train lengths.",
    )
    frequency_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP, description="Start and Stop (MHz) of Form Common CZ drive frequency."
    )
    frequency_points: int | None | Keep = Field(
        KEEP, description="Expts (count) under Form Common CZ drive frequency."
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Form Reps (count): attempted shots per sweep point in one round.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Form Rounds (count): acquisitions accumulated at each sweep point.",
    )


class ConditionalPhaseReport(RecipeReport):
    """Owner's unique π crossing and reason when there is none."""

    frequency_mhz: float | None = Field(
        default=None,
        description="Analyze frequency_mhz: common CZ drive frequency at the unique π crossing.",
    )
    frequency_reason: str | None = Field(
        default=None, description="Analyze frequency_reason for no frequency proposal."
    )


@recipe(
    "two_qubit_conditional_phase",
    description="Scan common CZ frequency, review a unique conditional π crossing for writeback and save.",
)
async def two_qubit_conditional_phase(
    session: RecipeSession, inputs: ConditionalPhaseInputs
) -> ConditionalPhaseReport:
    """Scan CZ frequency for the Ramsey role and review a unique π crossing.

    See [Select a frequency and apply](../v2/two_qubit/conditional_phase/guide.md#select-a-frequency-and-apply).
    Set ramsey_role (Form target), positive gate_count values, frequency_range
    in MHz and frequency_points (frequency.n → Expts). Omission keeps the Reset
    value. Null frequency_range clears both endpoints; other nulls go to the Form.
    Use overrides for pulses, readout and optional ordinary reset.
    Ask for selected Primary writeback only when there is a unique crossing.
    Save after a completed Run, even if analysis fails.
    """
    tab = session.open_tab(
        "two_qubit/conditional_phase",
        target=inputs.target.aliases() if inputs.target is not None else None,
        reuse=inputs.reuse_tab_id,
    )
    tab.set("target", inputs.ramsey_role)
    tab.set("gate_count", inputs.gate_count)
    if inputs.frequency_range is None:
        tab.set("frequency.start", None)
        tab.set("frequency.stop", None)
    elif not isinstance(inputs.frequency_range, Keep):
        start, stop = inputs.frequency_range
        tab.set("frequency.start", start)
        tab.set("frequency.stop", stop)
    tab.set("frequency.n", inputs.frequency_points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ConditionalPhaseAnalysis)
        if analysis.result is None:
            return ConditionalPhaseReport(analysis_failure=analysis.reason)

        crossing = analysis.result
        applied = False
        if crossing.frequency_mhz is not None:
            decision = await tab.ask_writeback(stage="primary")
            applied = apply_selected_writeback(tab, decision)

        return ConditionalPhaseReport(
            frequency_mhz=report_number(crossing.frequency_mhz),
            frequency_reason=crossing.frequency_reason,
            analysis_failure=crossing.frequency_reason,
            writeback_applied=applied,
        )
    finally:
        tab.save()


DEFINITION = two_qubit_conditional_phase
