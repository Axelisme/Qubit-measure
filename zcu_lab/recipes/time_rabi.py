"""Measure Rabi oscillations versus pulse length to calibrate rotation durations.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Literal

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2.twotone.rabi.len_rabi.gui import TimeRabiSummary

from .helpers import (
    FrequencyCalibration,
    choose_frequency,
    select_or_disable,
)


class TimeRabiInputs(RecipeInputs):
    """Inputs for time rabi; all units and null policies are declared per field."""

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

    decay: bool | None = Field(
        None,
        description="Fit decay envelope; null keeps the GUI analysis value. Analysis only.",
    )
    fit_phase: bool | None = Field(
        None,
        description="Fit phase offset; null keeps the GUI analysis value. Analysis only.",
    )


class TimeRabiReport(RecipeReport):
    """Time Rabi results and failure reason; cancellation returns no report."""

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

    points → expts sets the sweep sample count.
    Analysis controls: decay, fit_phase. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_recipe_tab("twotone/rabi/len_rabi", inputs, target_alias="qubit")
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select(("modules", "qub_pulse"), ref=inputs.drive_ref)
    tab.select_sources(inputs)

    edits: list[FieldEdit] = []
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
        edits.append(frequency.edit)
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
        edits.append(frequency.edit)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    edits.append(
        FieldEdit(path=("sweep", "length", "stop"), value=inputs.max_length_us)
    )
    edits.append(FieldEdit(path=("sweep", "length", "n"), value=inputs.points))
    if inputs.gain is not None:
        tab.select(("modules", "qub_pulse"), inline=True)
        edits.append(
            FieldEdit(path=("modules", "qub_pulse", "gain"), value=inputs.gain)
        )
    tab.apply_inputs(
        inputs, {("reps",): inputs.reps, ("rounds",): inputs.rounds}, edits=edits
    )

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=TimeRabiSummary,
            updates={
                key: value
                for key, value in {
                    "decay": inputs.decay,
                    "fit_phase": inputs.fit_phase,
                }.items()
                if value is not None
            },
        )
        if analysis.result is None:
            return TimeRabiReport(
                pi_length_us=None,
                pi_length_err_us=None,
                pi2_length_us=None,
                pi2_length_err_us=None,
                rabi_frequency_mhz=None,
                rabi_frequency_err_mhz=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return TimeRabiReport(
            pi_length_us=report_number(analysis.result.pi_len),
            pi_length_err_us=report_number(analysis.result.pi_len_err),
            pi2_length_us=report_number(analysis.result.pi2_len),
            pi2_length_err_us=report_number(analysis.result.pi2_len_err),
            rabi_frequency_mhz=report_number(analysis.result.rabi_f),
            rabi_frequency_err_mhz=report_number(analysis.result.rabi_f_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = time_rabi
