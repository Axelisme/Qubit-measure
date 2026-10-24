"""Measure excited-state relaxation versus delay to estimate T1.

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

from zcu_lab.v2.twotone.time_domain.t1.gui import T1Summary


class T1Inputs(RecipeInputs):
    """Inputs for t1; all units and null policies are declared per field."""

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
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Delay sweep stop in us; omission retains the Form value, null clears/disables it.",
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

    dual_exp: bool | None | Keep = Field(
        KEEP,
        description="Dual exponential; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    skip: int | None | Keep = Field(
        KEEP,
        description="Skip leading points; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class T1Report(RecipeReport):
    """T1 results and failure reason; cancellation returns no report."""

    t1_us: float | None = Field(
        description="Relaxation time in us; null means not estimable."
    )
    t1_err_us: float | None = Field(
        description="Relaxation uncertainty in us; null means not estimable."
    )
    t1b_us: float | None = Field(
        description="Second relaxation time in us; null without a second fit or when not estimable."
    )
    t1b_err_us: float | None = Field(
        description="Second relaxation uncertainty in us; null without a second fit or when not estimable."
    )


@recipe(
    "t1",
    description="Run one calibrated T1 delay sweep and save raw and Primary analysis.",
)
async def t1(session: RecipeSession, inputs: T1Inputs) -> T1Report:
    """Measure T1 from the decay of the qubit's excited-state population.

    After optional reset, apply a calibrated pi pulse, wait the swept delay, and
    read out the qubit. Fit the decay to estimate the relaxation
    time, and ask before applying the proposed T1; skipping writes nothing.

    points → expts sets the sweep sample count.
    Analysis controls: dual_exp, skip. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/t1",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
    tab.select("readout", inputs.readout_ref)
    tab.select("pi_pulse", inputs.pi_ref)
    tab.set("length.stop", inputs.max_delay_us)
    tab.set("length.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=T1Summary,
            updates={
                "dual_exp": inputs.dual_exp,
                "skip": inputs.skip,
            },
        )
        if analysis.result is None:
            return T1Report(
                t1_us=None,
                t1_err_us=None,
                t1b_us=None,
                t1b_err_us=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return T1Report(
            t1_us=report_number(analysis.result.t1),
            t1_err_us=report_number(analysis.result.t1_err),
            t1b_us=report_number(analysis.result.t1b),
            t1b_err_us=report_number(analysis.result.t1b_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t1
