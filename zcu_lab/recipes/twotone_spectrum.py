"""Qubit drive-frequency sweep, transition fit and reviewed calibration.

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

from zcu_lab.v2.twotone.freq.gui import TwotoneSpectrumSummary


class TwotoneSpectrumInputs(RecipeInputs):
    """Inputs for twotone spectrum; all units and null policies are declared per
    field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value; null clears/disables it. Sources are chosen by the Form.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Drive library key; omission retains the Form value; null clears/disables it. No pi calibration is required for this sweep.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Reset module option; omission retains the Form selection, null disables it.",
    )
    center_mhz: float = Field(description="Sweep center in MHz.")
    span_mhz: float = Field(
        gt=0,
        description="Full frequency span in MHz; endpoints are center minus/plus span/2.",
    )
    gain: float | None | Keep = Field(
        KEEP,
        description="Fixed pulse amplitude in native gain units; omission retains the Form value; null clears/disables it.",
    )
    pulse_length_us: float | None | Keep = Field(
        KEEP,
        description="Fixed drive pulse duration in us; omission retains the Form value; null clears/disables it.",
    )
    points: int = Field(
        ge=2,
        description="Sweep sample count; omission retains the Form value; null clears/disables it.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Acquisition repetition count; omission retains the Form value; null clears/disables it.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count; omission retains the Form value; null clears/disables it.",
    )

    model_type: Literal["lor", "sinc"] | None | Keep = Field(
        KEEP,
        description="Model type; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    plot_fit: bool | None | Keep = Field(
        KEEP,
        description="Plot fit; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
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
    Analysis controls: model_type, plot_fit. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/freq",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("modules.readout", inputs.readout_ref)
    tab.select("modules.reset", inputs.use_reset)
    tab.select("modules.qub_pulse", inputs.drive_ref)
    if not isinstance(inputs.gain, Keep) or not isinstance(
        inputs.pulse_length_us, Keep
    ):
        tab.command("modules.qub_pulse", "to_inline")
    if not isinstance(inputs.pulse_length_us, Keep):
        tab.command("modules.qub_pulse.waveform", "to_inline")
    tab.set("sweep.freq.start", inputs.center_mhz - inputs.span_mhz / 2)
    tab.set("sweep.freq.stop", inputs.center_mhz + inputs.span_mhz / 2)
    tab.set("sweep.freq.n", inputs.points)
    tab.set("modules.qub_pulse.gain", inputs.gain)
    tab.set("modules.qub_pulse.waveform.length", inputs.pulse_length_us)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=TwotoneSpectrumSummary,
            updates={
                "model_type": inputs.model_type,
                "plot_fit": inputs.plot_fit,
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
