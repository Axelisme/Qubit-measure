"""Measure Rabi oscillations versus pulse length to calibrate rotation durations."""

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
    retain_scalar,
    select_or_disable,
)


class TimeRabiInputs(FormModel):
    """Inputs for time rabi; all units and null policies are declared per field."""

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
    gain: float | None = Field(
        None,
        description="Fixed pulse amplitude in native gain units; null keeps the GUI value.",
    )
    max_length_us: float | None = Field(
        None,
        description="Pulse-length sweep stop in us; null keeps the GUI stop and start.",
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


class TimeRabiReport(FormModel):
    """Completed time rabi results; cancellation returns no report."""

    pi_length_us: float | None = Field(
        description="Pi pulse duration in us; null means not estimable."
    )
    pi_length_err_us: float | None = Field(
        description="Pi duration uncertainty in us; null means not estimable."
    )
    pi2_length_us: float | None = Field(
        description="Pi/2 pulse duration in us; null means not estimable."
    )
    pi2_length_err_us: float | None = Field(
        description="Pi/2 duration uncertainty in us; null means not estimable."
    )
    rabi_frequency_mhz: float | None = Field(
        description="Fitted Rabi oscillation frequency in MHz; null means not estimable."
    )
    rabi_frequency_err_mhz: float | None = Field(
        description="Rabi frequency uncertainty in MHz; null means not estimable."
    )
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )


class TimeRabiSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    pi_len: float | None = Field(
        description="Owner pi_len in us; null retains an unavailable estimate."
    )

    pi_len_err: float | None = Field(
        description="Owner pi_len_err in us; null retains an unavailable estimate."
    )

    pi2_len: float | None = Field(
        description="Owner pi2_len in us; null retains an unavailable estimate."
    )

    pi2_len_err: float | None = Field(
        description="Owner pi2_len_err in us; null retains an unavailable estimate."
    )

    rabi_f: float | None = Field(
        description="Owner rabi_f in MHz; null retains an unavailable estimate."
    )

    rabi_f_err: float | None = Field(
        description="Owner rabi_f_err in MHz; null retains an unavailable estimate."
    )


@recipe(
    "time_rabi",
    description="Run one length Rabi sweep and save raw data and Primary analysis.",
)
async def time_rabi(session: RecipeSession, inputs: TimeRabiInputs) -> TimeRabiReport:
    """Sweep pulse length at fixed gain to calibrate pi and pi/2 rotations.

    After optional reset, apply the drive pulse and read out the qubit at each
    length. Fit the Rabi oscillations to estimate pi and pi/2
    durations, and ask before applying the proposed calibration; skipping writes
    nothing.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Owner, acquisition, save and analysis errors
    propagate. Save raw data and completed artifacts once in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "twotone/rabi/len_rabi",
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
        session,
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
        session,
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
    tab.set(("sweep", "length", "stop"), inputs.max_length_us)
    tab.set(("sweep", "length", "n"), inputs.points)
    if inputs.gain is not None:
        tab.select(("modules", "qub_pulse"), inline=True)
        tab.set(("modules", "qub_pulse", "gain"), inputs.gain)
    else:
        retain_scalar(tab, ("modules", "qub_pulse", "gain"))
    tab.set(("reps",), inputs.reps)
    tab.set(("rounds",), inputs.rounds)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Analyze the acquisition with the GUI parameters.
        analysis = await tab.analyze("primary", result=TimeRabiSummary)

        # Write only the proposal items selected in the answer.
        decision = await tab.ask_writeback()
        if decision.accepted:
            tab.accept_writeback(decision.items)
        return TimeRabiReport(
            pi_length_us=report_number(analysis.result.pi_len),
            pi_length_err_us=report_number(analysis.result.pi_len_err),
            pi2_length_us=report_number(analysis.result.pi2_len),
            pi2_length_err_us=report_number(analysis.result.pi2_len_err),
            rabi_frequency_mhz=report_number(analysis.result.rabi_f),
            rabi_frequency_err_mhz=report_number(analysis.result.rabi_f_err),
            writeback_applied=decision.accepted,
        )
    finally:
        tab.save()


DEFINITION = time_rabi
