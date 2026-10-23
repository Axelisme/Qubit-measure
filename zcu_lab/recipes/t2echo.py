"""Measure spin-echo decay versus total delay to estimate refocused coherence.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Literal

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from .helpers import (
    FrequencyCalibration,
    apply_frequency,
    choose_frequency,
    report_number,
    require_pi,
    select_or_disable,
)


class T2EchoInputs(FormModel):
    """Inputs for t2echo; all units and null policies are declared per field."""

    target: str | None = Field(
        None,
        description="Qubit alias for this Reset; null uses the GUI aliases without changing focus.",
    )
    reuse_tab_id: str | None = Field(
        None,
        description="Idle matching tab to reset; null opens a new tab. May be combined with target.",
    )
    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the GUI selection. Frequency uses a valid captured library value, then resonator calibration.",
    )
    use_reset: str | Literal["off"] = Field(
        "off",
        description="Reset library key, or off to disable reset. Omission disables reset; null is rejected.",
    )
    pi_ref: str | None = Field(
        None,
        description="Calibrated pi-pulse library key; null keeps the GUI selection, which must be a valid reference.",
    )
    pi2_ref: str | None = Field(
        None,
        description="Calibrated pi/2-pulse library key; null keeps the GUI selection, which must be a valid reference.",
    )
    max_delay_us: float | None = Field(
        None,
        description="Total echo-delay stop in us, not half-delay; null keeps the GUI sweep.",
    )
    detune_ratio: float | None = Field(
        None,
        description="Final-pulse phase-ramp ratio, dimensionless; null keeps the GUI value.",
    )
    points: int | None = Field(
        None, description="Sweep sample count; null keeps the GUI count when optional."
    )
    reps: int | None = Field(
        None, description="Acquisition repetition count; null keeps the GUI value."
    )
    rounds: int | None = Field(
        None, description="Averaging round count; null keeps the GUI value."
    )


class T2EchoReport(FormModel):
    """T2 Echo results and failure reason; cancellation returns no report."""

    t2echo_us: float | None = Field(
        description="Refocused coherence time in us; null means not estimable."
    )
    t2echo_err_us: float | None = Field(
        description="Refocused coherence uncertainty in us; null means not estimable."
    )
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class T2EchoSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    t2e: float | None = Field(
        description="Owner t2e in us; null retains an unavailable estimate."
    )

    t2e_err: float | None = Field(
        description="Owner t2e_err in us; null retains an unavailable estimate."
    )


@recipe("t2echo", description="Run one calibrated Echo total-delay sweep.")
async def t2echo(session: RecipeSession, inputs: T2EchoInputs) -> T2EchoReport:
    """Measure refocused qubit coherence with a spin-echo delay sweep.

    After optional reset, apply pi/2, wait half the total delay, apply pi, wait
    the other half, then apply a phase-shifted pi/2 pulse and read out.
    Fit the decay to estimate echo T2, with optional fringe fitting.
    Ask before applying the proposed calibration, then save raw IQ and analysis.
    Skipping writes nothing.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Expected analysis failures return a reason in
    the report; other errors propagate. Save raw data and completed artifacts once
    in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "twotone/t2echo",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    missing: list[MissingParameter] = []
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
        calibration=FrequencyCalibration(
            path="$resonator.freq", required="readout_ref"
        ),
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.select(("modules", "readout"), inline=True)
        apply_frequency(tab, frequency.edit)
    if (
        parameter := require_pi(
            tab, ("modules", "pi_pulse"), inputs.pi_ref, required="pi_ref"
        )
    ) is not None:
        missing.append(parameter)
    if (
        parameter := require_pi(
            tab, ("modules", "pi2_pulse"), inputs.pi2_ref, required="pi2_ref"
        )
    ) is not None:
        missing.append(parameter)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    tab.set(("sweep", "length", "stop"), inputs.max_delay_us)
    tab.set(("sweep", "length", "n"), inputs.points)
    tab.set(("reps",), inputs.reps)
    tab.set(("rounds",), inputs.rounds)
    tab.set(("extras", "detune_ratio"), inputs.detune_ratio)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Analyze the acquisition with the GUI parameters.
        analysis = await tab.analyze("primary", result=T2EchoSummary)
        if analysis.result is None:
            return T2EchoReport(
                t2echo_us=None,
                t2echo_err_us=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        # Write only the proposal items selected in the answer.
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
