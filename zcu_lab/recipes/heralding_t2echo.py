"""Measure ground-heralded spin-echo decay versus total delay to estimate refocused coherence.

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
        description="Pulse-readout library key including classifier. Null passes to the Form owner.",
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
    description="Run ground-heralded Echo; save per-class shot sums and analysis.",
)
async def heralding_t2echo(
    session: RecipeSession, inputs: HeraldingT2EchoInputs
) -> HeraldingT2EchoReport:
    """Measure refocused qubit coherence with a spin-echo delay sweep.

    Store per-class shot sums; compute herald-g conditional IQ for fitting.
    Readout accepts circle, linear or centroid3. Herald states are g/e/other
    for circle (other is rejection), g/e for linear (no rejection), and g/e/f
    for centroid3 (f is not other). Circle requires a calibrated GE Post radius.
    Only shots classified as herald-g contribute to signal IQ and fitting.
    After optional reset and ground herald, apply pi/2, wait half the total delay, apply pi, wait
    the other half, then apply a phase-shifted pi/2 pulse and read out.
    Average only accepted signal shots; fit the decay with the ordinary Echo owner.
    No active reset or joint herald is performed. Fringe fitting remains optional.
    Ask before applying the proposed calibration, then save per-class shot sums and analysis.
    Skipping writes nothing.

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
