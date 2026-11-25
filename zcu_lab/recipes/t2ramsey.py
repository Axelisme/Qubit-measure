"""Measure Ramsey fringes versus delay to estimate free-evolution coherence.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    Overrides,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    ReuseTabId,
    SingleTarget,
    recipe,
    report_number,
)

from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import T2RamseySummary


class T2RamseyInputs(RecipeInputs):
    """Inputs for t2ramsey; all units and null policies are declared per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Select the Readout library module that integrates IQ.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Select Reset before the pulse sequence; null disables it.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Select π/2 pulse, the calibrated π/2 rotation.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Set Free-evolution delay › Stop (µs), the delay sweep endpoint.",
    )
    detune_ratio: float | None | Keep = Field(
        KEEP,
        description="Set Detune ratio (fringes/step) for the final-pulse phase ramp.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Set Free-evolution delay › Expts (count), including both endpoints.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), shots per point in each round.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), the acquisition rounds to average.",
    )

    fit_fringe: bool | None | Keep = Field(
        KEEP,
        description="Fit fringe. Analysis only; omitted keeps value, null passes to owner.",
    )
    fit_phase: bool | None | Keep = Field(
        KEEP,
        description="Fit phase offset (fringe only). Analysis only; omitted keeps value, null passes to owner.",
    )


class T2RamseyReport(RecipeReport):
    """T2 Ramsey results and failure reason; cancellation returns no report."""

    t2ramsey_us: float | None = Field(
        description="T₂* (µs) from the Analyze title; null means not estimable."
    )
    t2ramsey_err_us: float | None = Field(
        description="Free-evolution coherence uncertainty in µs; null means not estimable."
    )
    detune_mhz: float | None = Field(
        description="Analyze fringe frequency (MHz), not Applied detune; zero for decay-only fitting, null if not estimable."
    )
    fit_fringe: bool = Field(
        description="Whether the fringe model was fitted; false when analysis failed."
    )


@recipe("t2ramsey", description="Run one calibrated Ramsey delay sweep.")
async def t2ramsey(session: RecipeSession, inputs: T2RamseyInputs) -> T2RamseyReport:
    """Measure Ramsey fringes and their decay during free qubit evolution.

    After optional reset, apply pi/2, wait the swept delay, then apply a
    phase-shifted pi/2 pulse and read out. Analyze estimates T₂* from the
    decay envelope; the fringe model also returns the observed fringe frequency. Ask before
    applying the proposed calibration; skipping writes nothing. Analysis keeps
    the GUI parameters, which fit a free fringe phase by default. Turn off Fit
    phase offset in the tab to use a fixed phase instead.

    points → expts sets the sweep sample count.
    Analysis controls: fit_fringe, fit_phase. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/time_domain/t2ramsey",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
    tab.select("readout", inputs.readout_ref)
    tab.select("pi2_pulse", inputs.pi2_ref)
    tab.set("delay.stop", inputs.max_delay_us)
    tab.set("delay.n", inputs.points)
    tab.set("detune_ratio", inputs.detune_ratio)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
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
            fit_quality=(
                None
                if analysis.result.fit_quality is None
                else analysis.result.fit_quality["fit"]
            ),
            t2ramsey_us=report_number(analysis.result.t2r),
            t2ramsey_err_us=report_number(analysis.result.t2r_err),
            detune_mhz=report_number(analysis.result.detune),
            fit_fringe=analysis.result.fit_fringe,
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t2ramsey
