"""Measure T1 from paired IQ moments without a classifier.

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

from zcu_lab.v2.twotone.time_domain.t1_cov.core import T1CovAnalysis


class T1CovInputs(RecipeInputs):
    """Inputs for t1_cov. Omitted controls keep their Form or analysis values."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Single-tone pulse-readout library key.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Preparation reset library key. Null clears the selection.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi-pulse library key.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Delay sweep stop in us.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Sweep sample count.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Acquisition repetition count.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count.",
    )

    dual_exp: bool | None | Keep = Field(
        KEEP,
        description="Dual exponential. Analysis only; omitted keeps value, null passes to owner.",
    )
    skip: int | None | Keep = Field(
        KEEP,
        description="Skip leading points. Analysis only; omitted keeps value, null passes to owner.",
    )


class T1CovReport(RecipeReport):
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
    "t1_cov",
    description="Estimate T1 from two readouts per shot and save moment sums and Primary analysis.",
)
async def t1_cov(session: RecipeSession, inputs: T1CovInputs) -> T1CovReport:
    """Estimate T1 from the state-change probability between two IQ reads.

    Play optional reset, pre-readout, a calibrated pi pulse, the swept
    delay, and post-readout. Use one single-tone pulse readout for both
    reads. Save the paired second-moment sums, without shot classification.
    Ask before applying the T1 proposal. Skipping the proposal writes nothing.

    points → expts sets the sweep sample count; max_delay_us is in us.
    skip omits leading fit points; dual_exp fits two decay terms. Omission
    keeps the current value and null passes unchanged to the owner.
    To change the state count, set n_states in the tab's Analysis settings.
    Missing Form values stop the run; completed data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/t1_cov",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
    tab.select("readout", inputs.readout_ref)
    tab.select("pi_pulse", inputs.pi_ref)
    tab.set("delay.stop", inputs.max_delay_us)
    tab.set("delay.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=T1CovAnalysis,
            updates={
                "dual_exp": inputs.dual_exp,
                "skip": inputs.skip,
            },
        )
        if analysis.result is None:
            return T1CovReport(
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

        return T1CovReport(
            fit_quality=analysis.result.fit_quality["fit"],
            t1_us=report_number(analysis.result.t1),
            t1_err_us=report_number(analysis.result.t1_err),
            t1b_us=report_number(analysis.result.t1b),
            t1b_err_us=report_number(analysis.result.t1b_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t1_cov
