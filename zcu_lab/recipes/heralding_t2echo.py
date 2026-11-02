"""Measure self-calibrated heralding spin-echo decay versus total delay to estimate refocused coherence.

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


class HeraldingT2EchoInputs(RecipeInputs):
    """Inputs for heralding_t2echo; all units and null policies are declared per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse-readout library key without requiring a classifier. Null passes to the Form owner.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Preparation module option. Null passes to the Form owner.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi-pulse library key. Null passes to the Form owner.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi/2-pulse library key. Null passes to the Form owner.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Total echo-delay stop in us, not half-delay. Null passes to the Form owner.",
    )
    detune_ratio: float | None | Keep = Field(
        KEEP,
        description="Final-pulse phase-ramp ratio, dimensionless. Null passes to the Form owner.",
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

    fit_method: Literal["fringe", "decay"] | None | Keep = Field(
        KEEP,
        description="Fit method. Analysis only; omitted keeps value, null passes to owner.",
    )
    fit_phase: bool | None | Keep = Field(
        KEEP,
        description="Fit phase offset (fringe only). Analysis only; omitted keeps value, null passes to owner.",
    )


class HeraldingT2EchoReport(RecipeReport):
    """T2 Echo results and failure reason; cancellation returns no report."""

    t2echo_us: float | None = Field(
        description="Refocused coherence time in us; null means not estimable."
    )
    t2echo_err_us: float | None = Field(
        description="Refocused coherence uncertainty in us; null means not estimable."
    )


@recipe(
    "heralding_t2echo",
    description="Run self-calibrated heralding Echo; save second-moment sums and analysis.",
)
async def heralding_t2echo(
    session: RecipeSession, inputs: HeraldingT2EchoInputs
) -> HeraldingT2EchoReport:
    """Measure refocused qubit coherence with a spin-echo delay sweep.

    Store (N, 6, 6) herald/signal second-moment sums without a classifier.
    Analyze self-calibrates shared readout and fits only C = 1 - tr J, the
    probability of changing state. K[i,j] = P(i -> j) is a diagnostic, not a
    fitted curve. pi_hat is the post-herald population; decay during herald
    can reduce it relative to the original thermal population.
    This recipe has no n_states input. To change the state count, edit n_states
    in the completed tab's Analysis settings and rerun Analyze.
    The Analysis default is 2; second-order IQ supports at most 3 states.
    Three collinear centers fail identification; readout tails bias estimates.
    No feedback or active reset is added by heralding. Ask before applying
    the ordinary calibration proposals; skipping writes nothing.

    points → expts sets the sweep sample count.
    Analysis controls: fit_method, fit_phase. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "heralding/t2echo",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
    tab.select("readout", inputs.readout_ref)
    tab.select("pi_pulse", inputs.pi_ref)
    tab.select("pi2_pulse", inputs.pi2_ref)
    tab.set("delay.stop", inputs.max_delay_us)
    tab.set("delay.n", inputs.points)
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
            return HeraldingT2EchoReport(
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

        return HeraldingT2EchoReport(
            t2echo_us=report_number(analysis.result.t2e),
            t2echo_err_us=report_number(analysis.result.t2e_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = heralding_t2echo
