"""Fit each role's local Z phase and review the correction pair."""

from dataclasses import replace
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

from zcu_lab.v2.two_qubit.local_z.core import LocalZAnalysis, LocalZFit

from .two_qubit_helpers import TwoQubitTarget, apply_selected_writeback


class LocalZInputs(RecipeInputs):
    """Four-role target and gate-count scan. Overrides apply after named edits."""

    target: TwoQubitTarget | None = Field(
        default=None,
        description="Role bindings for this Reset. Omitted or null roles keep the current aliases.",
    )
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    gate_count: list[Annotated[int, Field(strict=True, ge=0)]] | None | Keep = Field(
        KEEP,
        description="Form CZ gates per shot (count): consecutive nonnegative train lengths, including zero.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Form Reps (count): attempted shots per gate count and projection in one round.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Form Rounds (count): acquisitions accumulated at each sweep point.",
    )


class LocalZReport(RecipeReport):
    """Owner's phase-line fits by role, with no shared fit quality."""

    qubit1: LocalZFit | None = Field(
        default=None,
        description="Analyze fits[0]: Qubit1 phase slope, opposite-sign correction, fit quality or failure reason.",
    )
    qubit2: LocalZFit | None = Field(
        default=None,
        description="Analyze fits[1]: Qubit2 phase slope, opposite-sign correction, fit quality or failure reason.",
    )


def _report_fit(fit: LocalZFit) -> LocalZFit:
    return replace(
        fit,
        intercept_deg=report_number(fit.intercept_deg),
        slope_deg=report_number(fit.slope_deg),
        correction_deg=report_number(fit.correction_deg),
        slope_error_deg=report_number(fit.slope_error_deg),
    )


@recipe(
    "two_qubit_local_z",
    description="Fit local Z phase for Qubit1 and Qubit2, review the correction pair and save.",
)
async def two_qubit_local_z(
    session: RecipeSession, inputs: LocalZInputs
) -> LocalZReport:
    """Sweep CZ gate counts and fit one phase line in degrees for each role.

    See [How to read the plots](../v2/two_qubit/local_z/guide.md#怎麼讀圖).
    Set gate_count as a list of consecutive nonnegative integers. Omission
    keeps the Reset value; explicit null passes to the Form. Use overrides for
    pulses, readout and reset. Ask about the Primary correction pair only if
    both fits give corrections. Save after a completed Run, even without a pair.
    """
    tab = session.open_tab(
        "two_qubit/local_z",
        target=inputs.target.aliases() if inputs.target is not None else None,
        reuse=inputs.reuse_tab_id,
    )
    tab.set("gate_count", inputs.gate_count)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=LocalZAnalysis)
        if analysis.result is None:
            return LocalZReport(analysis_failure=analysis.reason)

        qubit1, qubit2 = map(_report_fit, analysis.result.fits)
        applied = False
        if qubit1.correction_deg is not None and qubit2.correction_deg is not None:
            decision = await tab.ask_writeback(stage="primary")
            applied = apply_selected_writeback(tab, decision)

        return LocalZReport(qubit1=qubit1, qubit2=qubit2, writeback_applied=applied)
    finally:
        tab.save()


DEFINITION = two_qubit_local_z
