"""Qubit drive-frequency sweep, transition fit and reviewed calibration.

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

from zcu_lab.v2.twotone.freq.gui import TwotoneSpectrumSummary

from .helpers import (
    FrequencyCalibration,
    FrequencySweepCalibration,
    FrequencySweepRange,
    choose_frequency,
    frequency_sweep_edits,
    select_or_disable,
)


class TwotoneSpectrumInputs(RecipeInputs):
    """Inputs for twotone spectrum; all units and null policies are declared per
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

    model_type: Literal["lor", "sinc"] | None = Field(
        None,
        description="Model type; null keeps the GUI analysis value. Analysis only.",
    )
    plot_fit: bool | None = Field(
        None, description="Plot fit; null keeps the GUI analysis value. Analysis only."
    )


class TwotoneSpectrumReport(RecipeReport):
    """Twotone spectrum results and failure reason; cancellation returns no report."""

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

    points → expts sets the sweep sample count.
    Analysis controls: model_type, plot_fit. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_recipe_tab("twotone/freq", inputs, target_alias="qubit")
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select(("modules", "qub_pulse"), ref=inputs.drive_ref)
    tab.select_sources(inputs)

    edits: list[FieldEdit] = []
    missing: list[MissingParameter] = []
    sweep = frequency_sweep_edits(
        tab,
        ("sweep", "freq"),
        calibration=FrequencySweepCalibration("$qubit.freq", "$qubit.kappa", 3.0),
        scan=FrequencySweepRange(
            center_mhz=inputs.center_mhz, span_mhz=inputs.span_mhz, expts=inputs.points
        ),
    )
    missing.extend(sweep.missing)
    edits.extend(sweep.edits)
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
    if inputs.gain is not None or inputs.pulse_length_us is not None:
        tab.select(("modules", "qub_pulse"), inline=True)
        edits.append(
            FieldEdit(path=("modules", "qub_pulse", "gain"), value=inputs.gain)
        )
    if inputs.pulse_length_us is not None:
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
            result=TwotoneSpectrumSummary,
            updates={
                key: value
                for key, value in {
                    "model_type": inputs.model_type,
                    "plot_fit": inputs.plot_fit,
                }.items()
                if value is not None
            },
        )
        if analysis.result is None:
            return TwotoneSpectrumReport(
                frequency_mhz=None,
                frequency_err_mhz=None,
                linewidth_mhz=None,
                linewidth_err_mhz=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

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
