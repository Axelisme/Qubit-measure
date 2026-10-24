"""Measure spin-echo decay versus total delay to estimate refocused coherence.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Literal

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoSummary


class T2EchoInputs(RecipeInputs):
    """Inputs for t2echo; all units and null policies are declared per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value, null clears/disables it.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Preparation module option; omission retains the Form value, null clears/disables it.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi-pulse library key; omission retains the Form value, null clears/disables it.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi/2-pulse library key; omission retains the Form value, null clears/disables it.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Total echo-delay stop in us, not half-delay; omission retains the Form value, null clears/disables it.",
    )
    detune_ratio: float | None | Keep = Field(
        KEEP,
        description="Final-pulse phase-ramp ratio, dimensionless; omission retains the Form value, null clears/disables it.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Sweep sample count; omission retains the Form value, null clears/disables it.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Acquisition repetition count; omission retains the Form value, null clears/disables it.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count; omission retains the Form value, null clears/disables it.",
    )

    fit_method: Literal["fringe", "decay"] | None | Keep = Field(
        KEEP,
        description="Fit method; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    fit_phase: bool | None | Keep = Field(
        KEEP,
        description="Fit phase offset (fringe only); omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class T2EchoReport(RecipeReport):
    """T2 Echo results and failure reason; cancellation returns no report."""

    t2echo_us: float | None = Field(
        description="Refocused coherence time in us; null means not estimable."
    )
    t2echo_err_us: float | None = Field(
        description="Refocused coherence uncertainty in us; null means not estimable."
    )


@recipe("t2echo", description="Run one calibrated Echo total-delay sweep.")
async def t2echo(session: RecipeSession, inputs: T2EchoInputs) -> T2EchoReport:
    """Measure refocused qubit coherence with a spin-echo delay sweep.

    After optional reset, apply pi/2, wait half the total delay, apply pi, wait
    the other half, then apply a phase-shifted pi/2 pulse and read out.
    Fit the decay to estimate echo T2, with optional fringe fitting.
    Ask before applying the proposed calibration, then save raw IQ and analysis.
    Skipping writes nothing.

    points → expts sets the sweep sample count.
    Analysis controls: fit_method, fit_phase. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/t2echo",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
    tab.select("readout", inputs.readout_ref)
    tab.select("pi_pulse", inputs.pi_ref)
    tab.select("pi2_pulse", inputs.pi2_ref)
    tab.set("length.stop", inputs.max_delay_us)
    tab.set("length.n", inputs.points)
    tab.set("detune_ratio", inputs.detune_ratio)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=T2EchoSummary,
            updates={
                "fit_method": inputs.fit_method,
                "fit_phase": inputs.fit_phase,
            },
        )
        if analysis.result is None:
            return T2EchoReport(
                t2echo_us=None,
                t2echo_err_us=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return T2EchoReport(
            t2echo_us=report_number(analysis.result.t2e),
            t2echo_err_us=report_number(analysis.result.t2e_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t2echo
