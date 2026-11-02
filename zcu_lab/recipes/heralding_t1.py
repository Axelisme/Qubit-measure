"""Measure single-qubit T1 after ground-herald post-selection.

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


class HeraldingT1Inputs(RecipeInputs):
    """Inputs for heralding_t1; all units and null policies are declared per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse-readout library key without requiring a classifier. Null passes to the Form owner.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi-pulse library key. Null passes to the Form owner.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Delay sweep stop in us. Null passes to the Form owner.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Sweep sample count. Null passes to the Form owner.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Acquisition repetition count. Null passes to the Form owner.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count. Null passes to the Form owner.",
    )

    dual_exp: bool | None | Keep = Field(
        KEEP,
        description="Dual exponential. Analysis only; omitted keeps value, null passes to owner.",
    )
    skip: int | None | Keep = Field(
        KEEP,
        description="Skip leading points. Analysis only; omitted keeps value, null passes to owner.",
    )


class HeraldingT1Report(RecipeReport):
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
    "heralding_t1",
    description="Run single-qubit self-calibrated heralding T1; save second-moment sums and analysis.",
)
async def heralding_t1(
    session: RecipeSession, inputs: HeraldingT1Inputs
) -> HeraldingT1Report:
    """Measure T1 from the decay of the qubit's excited-state population.

    Store (N, 6, 6) herald/signal second-moment sums without a classifier.
    Analyze self-calibrates shared readout and fits only C = 1 - tr J, the
    probability of changing state. K[i,j] = P(i -> j) is a diagnostic, not a
    fitted curve. pi_hat is the post-herald population; decay during herald
    can reduce it relative to the original thermal population.
    Analysis overrides select n_states=2 (default) or 3, the second-order limit.
    Three collinear centers fail identification; readout tails bias estimates.
    No feedback or active reset is added by heralding. Ask before applying
    the ordinary calibration proposals; skipping writes nothing.

    points → expts sets the sweep sample count.
    Analysis controls: dual_exp, skip. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "heralding/t1",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("pi_pulse", inputs.pi_ref)
    tab.set("delay.stop", inputs.max_delay_us)
    tab.set("delay.n", inputs.points)
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
            return HeraldingT1Report(
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

        return HeraldingT1Report(
            t1_us=report_number(analysis.result.t1),
            t1_err_us=report_number(analysis.result.t1_err),
            t1b_us=report_number(analysis.result.t1b),
            t1b_err_us=report_number(analysis.result.t1b_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = heralding_t1
