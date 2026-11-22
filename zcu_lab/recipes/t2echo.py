"""Measure spin-echo decay versus total delay to estimate refocused coherence.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Literal

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

from zcu_lab.v2.twotone.time_domain.t2echo.gui import T2EchoSummary


class T2EchoInputs(RecipeInputs):
    """Inputs for t2echo; all units and null policies are declared per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Select Readout by calibrated library key; null fails at module selection before overrides.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Select Reset before the first π/2 pulse; null skips it.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Select the refocusing Pi Pulse by library key; null fails at module selection before overrides.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Select Pi/2 Pulse by library key; null fails at module selection before overrides.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Set Total echo delay › Stop (µs); null leaves a required blank at Run readiness.",
    )
    detune_ratio: float | None | Keep = Field(
        KEEP,
        description="Set Detune ratio (fringes/step); null leaves a required blank at Run readiness.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Set Total echo delay › Expts (count); null leaves a required blank at Run readiness.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), shots per delay each round; null leaves a required blank at Run readiness.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), the sweep limit; null leaves a required blank at Run readiness.",
    )

    fit_method: Literal["fringe", "decay"] | None | Keep = Field(
        KEEP,
        description="Set Fit method to decay or fringe; omission keeps the Analyze value and null fails at Analyze.",
    )
    fit_phase: bool | None | Keep = Field(
        KEEP,
        description="Set Fit phase offset (fringe only); omission keeps the value, null fails at Analyze.",
    )


class T2EchoReport(RecipeReport):
    """T2 Echo results and failure reason; cancellation returns no report."""

    t2echo_us: float | None = Field(
        description="T₂ Echo (µs), the fitted envelope decay time; null means unavailable."
    )
    t2echo_err_us: float | None = Field(
        description="T₂ Echo fit standard error (µs); null means unavailable."
    )


@recipe("t2echo", description="Run one calibrated Echo total-delay sweep.")
async def t2echo(session: RecipeSession, inputs: T2EchoInputs) -> T2EchoReport:
    """Measure refocused qubit coherence with a spin-echo delay sweep.

    After optional reset, apply pi/2, wait half the total delay, apply pi, wait
    the other half, then apply a phase-shifted pi/2 pulse and read out.
    Fit the decay to estimate echo T2, with optional fringe fitting.
    Ask before applying the proposed T₂ Echo calibration. Skipping writeback
    leaves the calibration unchanged. Save mean IQ and any available analysis
    even when the fit fails.

    points → expts sets Total echo delay › Expts (count). Omitted inputs leave
    their Form or Analyze values unchanged. Explicit null follows the field's
    selection, readiness or Analyze rule in the input schema.
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
            fit_quality=(
                None
                if analysis.result.fit_quality is None
                else analysis.result.fit_quality["fit"]
            ),
            t2echo_us=report_number(analysis.result.t2e),
            t2echo_err_us=report_number(analysis.result.t2e_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = t2echo
