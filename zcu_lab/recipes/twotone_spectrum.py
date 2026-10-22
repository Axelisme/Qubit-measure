"""Qubit drive-frequency sweep, transition fit and reviewed calibration."""

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
    FrequencySweepCalibration,
    FrequencySweepRange,
    apply_frequency,
    choose_frequency,
    report_number,
    retain_scalar,
    select_or_disable,
    set_frequency_sweep,
)


class TwotoneSpectrumInputs(FormModel):
    """Inputs for twotone spectrum; all units and null policies are declared per
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
    center_mhz: float = Field(description="Sweep center in MHz.")
    span_mhz: float = Field(
        description="Full frequency span in MHz; endpoints are center minus/plus span/2."
    )
    gain: float | None = Field(
        None,
        description="Fixed pulse amplitude in native gain units; null keeps the GUI value.",
    )
    pulse_length_us: float | None = Field(
        None, description="Fixed drive pulse duration in us; null keeps the GUI value."
    )
    points: int = Field(
        description="Sweep sample count; null keeps the GUI count when optional."
    )
    reps: int | None = Field(
        None, description="Acquisition repetition count; null keeps the GUI value."
    )
    rounds: int | None = Field(
        None, description="Averaging round count; null keeps the GUI value."
    )


class TwotoneSpectrumReport(FormModel):
    """Completed twotone spectrum results; cancellation returns no report."""

    frequency_mhz: float | None = Field(
        description="Fitted transition frequency in MHz; null means not estimable."
    )
    frequency_err_mhz: float | None = Field(
        description="Transition frequency uncertainty in MHz; null means not estimable."
    )
    linewidth_mhz: float | None = Field(
        description="Fitted full linewidth in MHz; null means not estimable."
    )
    linewidth_err_mhz: float | None = Field(
        description="Linewidth uncertainty in MHz; null means not estimable."
    )
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )


class TwotoneSpectrumSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    freq: float | None = Field(
        description="Owner freq in MHz; null retains an unavailable estimate."
    )

    freq_err: float | None = Field(
        description="Owner freq_err in MHz; null retains an unavailable estimate."
    )

    fwhm: float | None = Field(
        description="Owner fwhm in MHz; null retains an unavailable estimate."
    )

    fwhm_err: float | None = Field(
        description="Owner fwhm_err in MHz; null retains an unavailable estimate."
    )


@recipe(
    "twotone_spectrum",
    description="Run one two-tone spectrum and save raw data and Primary analysis.",
)
async def twotone_spectrum(
    session: RecipeSession, inputs: TwotoneSpectrumInputs
) -> TwotoneSpectrumReport:
    """Sweep qubit drive frequency to locate a transition.

    Apply optional reset, a fixed-length/gain qubit pulse, then readout. Primary
    fits the averaged IQ response to estimate transition frequency and
    linewidth. Ask before applying it, then save raw data and analysis.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Owner, acquisition, save and analysis errors
    propagate. Save raw data and completed artifacts once in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "twotone/freq",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select(("modules", "qub_pulse"), ref=inputs.drive_ref)
    missing: list[MissingParameter] = []
    missing.extend(
        set_frequency_sweep(
            tab,
            ("sweep", "freq"),
            calibration=FrequencySweepCalibration("$qubit.freq", "$qubit.kappa", 3.0),
            session=session,
            scan=FrequencySweepRange(
                center_mhz=inputs.center_mhz,
                span_mhz=inputs.span_mhz,
                expts=inputs.points,
            ),
        )
    )
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
    if inputs.gain is not None or inputs.pulse_length_us is not None:
        tab.select(("modules", "qub_pulse"), inline=True)
        tab.set(("modules", "qub_pulse", "gain"), inputs.gain)
    else:
        retain_scalar(tab, ("modules", "qub_pulse", "gain"))
    if inputs.pulse_length_us is not None:
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
        analysis = await tab.analyze("primary", result=TwotoneSpectrumSummary)

        # Write only the proposal items selected in the answer.
        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)
        return TwotoneSpectrumReport(
            frequency_mhz=report_number(analysis.result.freq),
            frequency_err_mhz=report_number(analysis.result.freq_err),
            linewidth_mhz=report_number(analysis.result.fwhm),
            linewidth_err_mhz=report_number(analysis.result.fwhm_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = twotone_spectrum
