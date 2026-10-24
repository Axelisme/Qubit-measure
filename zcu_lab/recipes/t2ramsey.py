"""Measure Ramsey fringes versus delay to estimate free-evolution coherence.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

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

from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseySummary


class T2RamseyInputs(RecipeInputs):
    """Inputs for t2ramsey; all units and null policies are declared per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value, null clears/disables it.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Preparation module option; omission retains the Form value, null clears/disables it.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi/2-pulse library key; omission retains the Form value, null clears/disables it.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Delay sweep stop in us; omission retains the Form value, null clears/disables it.",
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

    fit_fringe: bool | None | Keep = Field(
        KEEP,
        description="Fit fringe; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    fit_phase: bool | None | Keep = Field(
        KEEP,
        description="Fit phase offset (fringe only); omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class T2RamseyReport(RecipeReport):
    """T2 Ramsey results and failure reason; cancellation returns no report."""

    t2ramsey_us: float | None = Field(
        description="Free-evolution coherence time in us; null means not estimable."
    )
    t2ramsey_err_us: float | None = Field(
        description="Free-evolution coherence uncertainty in us; null means not estimable."
    )
    detune_mhz: float | None = Field(
        description="Fitted fringe frequency in MHz; zero for decay-only fitting; null means not estimable."
    )
    fit_fringe: bool = Field(
        description="Whether the fringe model was fitted; false when analysis failed."
    )


@recipe("t2ramsey", description="Run one calibrated Ramsey delay sweep.")
async def t2ramsey(session: RecipeSession, inputs: T2RamseyInputs) -> T2RamseyReport:
    """Measure Ramsey fringes and their decay during free qubit evolution.

    After optional reset, apply pi/2, wait the swept delay, then apply a
    phase-shifted pi/2 pulse and read out. Analyze before fitting the decay
    to estimate Ramsey T2; fringe fitting also estimates detuning. Ask before
    applying the proposed calibration; skipping writes nothing. Analysis keeps
    the GUI parameters, which fit a free fringe phase by default. Turn off Fit
    phase offset in the tab to use a fixed phase instead.

    points → expts sets the sweep sample count.
    Analysis controls: fit_fringe, fit_phase. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/t2ramsey",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
    tab.select("readout", inputs.readout_ref)
    tab.select("pi2_pulse", inputs.pi2_ref)
    tab.set("length.stop", inputs.max_delay_us)
    tab.set("length.n", inputs.points)
    tab.set("extras.detune_ratio", inputs.detune_ratio)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=T2RamseySummary,
            updates={
                "fit_fringe": inputs.fit_fringe,
                "fit_phase": inputs.fit_phase,
            },
        )
        if analysis.result is None:
            return T2RamseyReport(
                t2ramsey_us=None,
                t2ramsey_err_us=None,
                detune_mhz=None,
                fit_fringe=False,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return T2RamseyReport(
            t2ramsey_us=report_number(analysis.result.t2r),
            t2ramsey_err_us=report_number(analysis.result.t2r_err),
            detune_mhz=report_number(analysis.result.detune),
            fit_fringe=analysis.result.fit_fringe,
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t2ramsey
