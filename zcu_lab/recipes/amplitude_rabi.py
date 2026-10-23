"""Measure Rabi oscillations versus drive gain to calibrate rotation amplitudes.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from math import isfinite
from typing import Literal

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.form_adapter.sweep import LinRange
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
    retain_scalar,
    select_or_disable,
)


def _check_gain_endpoints(gain_range: list[float] | None) -> None:
    """Require two finite native gain endpoints when a range is supplied."""
    if gain_range is not None and (
        len(gain_range) != 2 or not all(isfinite(value) for value in gain_range)
    ):
        raise ValueError("gain_range must contain exactly two finite endpoints")


class AmplitudeRabiInputs(FormModel):
    """Inputs for amplitude rabi; all units and null policies are declared per
    field."""

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
    drive_ref: str | None = Field(
        None,
        description="Drive library key; null keeps the GUI selection. No pi calibration is required for this sweep.",
    )
    use_reset: str | Literal["off"] = Field(
        "off",
        description="Reset library key, or off to disable reset. Omission disables reset; null is rejected.",
    )
    frequency_mhz: float | None = Field(
        None,
        description="Fixed pulse frequency in MHz. Explicit value wins, then a valid captured library frequency, then calibration.",
    )
    pulse_length_us: float | None = Field(
        None, description="Fixed drive pulse duration in us; null keeps the GUI value."
    )
    gain_range: list[float] | None = Field(
        None, description="Two native gain endpoints; null keeps the GUI range."
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


class AmplitudeRabiReport(FormModel):
    """Amplitude Rabi results and failure reason; cancellation returns no report."""

    pi_gain: float | None = Field(
        description="Pi rotation gain in native amplitude units; null means not estimable."
    )
    pi_gain_err: float | None = Field(
        description="Uncertainty of pi rotation gain; null means not estimable."
    )
    pi2_gain: float | None = Field(
        description="Pi/2 rotation gain in native amplitude units; null means not estimable."
    )
    pi2_gain_err: float | None = Field(
        description="Uncertainty of pi/2 rotation gain; null means not estimable."
    )
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class AmplitudeRabiSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    pi_gain: float | None = Field(
        description="Owner pi_gain in gain; null retains an unavailable estimate."
    )

    pi_gain_err: float | None = Field(
        description="Owner pi_gain_err in gain; null retains an unavailable estimate."
    )

    pi2_gain: float | None = Field(
        description="Owner pi2_gain in gain; null retains an unavailable estimate."
    )

    pi2_gain_err: float | None = Field(
        description="Owner pi2_gain_err in gain; null retains an unavailable estimate."
    )


@recipe(
    "amplitude_rabi",
    description="Run one gain Rabi sweep and save raw data and Primary analysis.",
)
async def amplitude_rabi(
    session: RecipeSession, inputs: AmplitudeRabiInputs
) -> AmplitudeRabiReport:
    """Sweep drive gain at fixed pulse length to calibrate pi and pi/2 rotations.

    After optional reset, apply the drive pulse and read out the qubit at each
    gain. Fit the Rabi oscillations to estimate pi and pi/2 gains,
    and ask before applying the proposed calibration; skipping writes nothing.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Expected analysis failures return a reason in
    the report; other errors propagate. Save raw data and completed artifacts once
    in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    _check_gain_endpoints(inputs.gain_range)
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "twotone/rabi/amp_rabi",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select(("modules", "qub_pulse"), ref=inputs.drive_ref)
    missing: list[MissingParameter] = []
    frequency = choose_frequency(
        tab,
        ("modules", "qub_pulse", "freq"),
        inputs.frequency_mhz,
        calibration=FrequencyCalibration(path="$qubit.freq", required="frequency_mhz"),
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.select(("modules", "qub_pulse"), inline=True)
        apply_frequency(tab, frequency.edit)
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
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    if inputs.gain_range is not None and inputs.points is not None:
        tab.set(
            ("sweep", "gain"),
            LinRange(
                start=inputs.gain_range[0], stop=inputs.gain_range[1], n=inputs.points
            ),
        )
    else:
        tab.set(
            ("sweep", "gain", "start"),
            inputs.gain_range[0] if inputs.gain_range is not None else None,
        )
        tab.set(
            ("sweep", "gain", "stop"),
            inputs.gain_range[1] if inputs.gain_range is not None else None,
        )
        tab.set(("sweep", "gain", "n"), inputs.points)
    if inputs.pulse_length_us is not None:
        tab.select(("modules", "qub_pulse"), inline=True)
        tab.select(("modules", "qub_pulse", "waveform"), inline=True)
        tab.set(("modules", "qub_pulse", "waveform", "length"), inputs.pulse_length_us)
    else:
        retain_scalar(tab, ("modules", "qub_pulse", "waveform", "length"))
    tab.set(("reps",), inputs.reps)
    tab.set(("rounds",), inputs.rounds)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Analyze the acquisition with the GUI parameters.
        analysis = await tab.analyze("primary", result=AmplitudeRabiSummary)
        if analysis.result is None:
            return AmplitudeRabiReport(
                pi_gain=None,
                pi_gain_err=None,
                pi2_gain=None,
                pi2_gain_err=None,
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
        return AmplitudeRabiReport(
            pi_gain=report_number(analysis.result.pi_gain),
            pi_gain_err=report_number(analysis.result.pi_gain_err),
            pi2_gain=report_number(analysis.result.pi2_gain),
            pi2_gain_err=report_number(analysis.result.pi2_gain_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = amplitude_rabi
