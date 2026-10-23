"""Measure Rabi oscillations versus drive gain to calibrate rotation amplitudes.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from math import isfinite
from typing import Literal

from pydantic import Field
from zcu_tools.gui.form_adapter.sweep import LinRange
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

from zcu_lab.v2.twotone.rabi.amp_rabi.gui import AmplitudeRabiSummary

from .helpers import (
    FrequencyCalibration,
    choose_frequency,
    select_or_disable,
)


def _check_gain_endpoints(gain_range: list[float] | None) -> None:
    """Require two finite native gain endpoints when a range is supplied."""
    if gain_range is not None and (
        len(gain_range) != 2 or not all(isfinite(value) for value in gain_range)
    ):
        raise ValueError("gain_range must contain exactly two finite endpoints")


class AmplitudeRabiInputs(RecipeInputs):
    """Inputs for amplitude rabi; all units and null policies are declared per
    field."""

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

    skip: int | None = Field(
        None,
        description="Skip points; null keeps the GUI analysis value. Analysis only.",
    )


class AmplitudeRabiReport(RecipeReport):
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

    points → expts sets the sweep sample count.
    Analysis controls: skip. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    _check_gain_endpoints(inputs.gain_range)
    tab = session.open_recipe_tab("twotone/rabi/amp_rabi", inputs, target_alias="qubit")
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
    if inputs.gain_range is not None and inputs.points is not None:
        edits.append(
            FieldEdit(
                path=("sweep", "gain"),
                value=LinRange(
                    start=inputs.gain_range[0],
                    stop=inputs.gain_range[1],
                    n=inputs.points,
                ),
            )
        )
    else:
        edits.append(
            FieldEdit(
                path=("sweep", "gain", "start"),
                value=inputs.gain_range[0] if inputs.gain_range is not None else None,
            )
        )
        edits.append(
            FieldEdit(
                path=("sweep", "gain", "stop"),
                value=inputs.gain_range[1] if inputs.gain_range is not None else None,
            )
        )
        edits.append(FieldEdit(path=("sweep", "gain", "n"), value=inputs.points))
    if inputs.pulse_length_us is not None:
        tab.select(("modules", "qub_pulse"), inline=True)
        tab.select(("modules", "qub_pulse", "waveform"), inline=True)
        edits.append(
            FieldEdit(
                path=("modules", "qub_pulse", "waveform", "length"),
                value=inputs.pulse_length_us,
            )
        )
    tab.apply_inputs(
        inputs, {("reps",): inputs.reps, ("rounds",): inputs.rounds}, edits=edits
    )
    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=AmplitudeRabiSummary,
            updates={
                key: value
                for key, value in {"skip": inputs.skip}.items()
                if value is not None
            },
        )
        if analysis.result is None:
            return AmplitudeRabiReport(
                pi_gain=None,
                pi_gain_err=None,
                pi2_gain=None,
                pi2_gain_err=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )
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
