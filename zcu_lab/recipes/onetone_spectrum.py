"""Resonator frequency sweep, spectral fit and reviewed calibration.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Annotated, Literal

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
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2.onetone.freq.gui import OnetoneSpectrumSummary


class OnetoneSpectrumInputs(RecipeInputs):
    """Inputs for onetone spectrum; all units and null policies are declared per
    field."""

    target: Annotated[
        SingleTarget,
        Field(
            description="Qubit binding used to resolve its resonator; omit or pass null to keep the current alias."
        ),
    ]
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Select the Readout pulse and ADC window by library key; omit to retain the Form selection, or pass null to the Form.",
    )
    center_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Readout RF frequency › Start and Stop (MHz) around this absolute RF center; provide span_mhz too. Omit both to retain the sweep; null clears both endpoints.",
    )
    span_mhz: Annotated[float, Field(gt=0)] | None | Keep = Field(
        KEEP,
        description="Set Readout RF frequency › Start and Stop (MHz) to center_mhz minus/plus half this full span. Omit both to retain the sweep; null clears both endpoints.",
    )
    gain: float | None | Keep = Field(
        KEEP,
        description="Set Readout › Gain (a.u.), the fraction of DAC full-scale pulse amplitude; omit to retain it, or pass null to the Form.",
    )
    points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP,
        description="Set Readout RF frequency › Expts (count); omit to retain it, or pass null to the Form.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), shots per frequency per round; omit to retain it, or pass null to the Form.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), repeats of the sweep for averaging; omit to retain it, or pass null to the Form.",
    )

    model_type: Literal["hm", "t", "auto"] | None | Keep = Field(
        KEEP,
        description="Set Analyze › Resonance model: hm for a hanger notch, t for a transmission peak, or auto to choose from the measured magnitude; omit to retain it, or pass null to Analyze.",
    )
    fit_bg_amp_slope: bool | None | Keep = Field(
        KEEP,
        description="Set Analyze › Fit amplitude background slope to allow a frequency-dependent amplitude envelope; omit to retain it, or pass null to Analyze.",
    )
    fit_bg_phase_curvature: bool | None | Keep = Field(
        KEEP,
        description="Set Analyze › Fit phase background curvature to allow a quadratic phase term; omit to retain it, or pass null to Analyze.",
    )
    edelay: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Electrical delay seed (µs; blank = auto); omit to retain it or pass null to use a matching saved calibration or automatic branch search.",
    )


class OnetoneSpectrumReport(RecipeReport):
    """Onetone spectrum results and failure reason; cancellation returns no report."""

    frequency_mhz: float | None = Field(
        description="Resonator frequency (MHz) from the fit; null means not estimable."
    )
    linewidth_mhz: float | None = Field(
        description="Linewidth (FWHM) (MHz) from the fit; null means not estimable."
    )


@recipe(
    "onetone_spectrum",
    description="Sweep the resonator readout RF frequency, fit the complex IQ spectrum and ask before writing the calibration; save the acquired data and analysis.",
)
async def onetone_spectrum(
    session: RecipeSession, inputs: OnetoneSpectrumInputs
) -> OnetoneSpectrumReport:
    """Sweep readout frequency to locate the resonator response.

    At each frequency, apply the readout pulse and average its IQ signal.
    Primary fits the spectrum to estimate resonator frequency and linewidth.
    Ask before applying the calibration, then save raw data and analysis.

    points → expts sets the sweep sample count.
    Analysis controls: model_type, fit_bg_amp_slope, fit_bg_phase_curvature, edelay. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "onetone/freq",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    if not isinstance(inputs.gain, Keep):
        tab.command("readout", "to_inline")
    if isinstance(inputs.center_mhz, Keep) != isinstance(inputs.span_mhz, Keep):
        missing = "span_mhz" if isinstance(inputs.span_mhz, Keep) else "center_mhz"
        raise RecipeNeedsParameters(
            (MissingParameter(missing, "Supply center_mhz and span_mhz together"),)
        )
    if inputs.center_mhz is None or inputs.span_mhz is None:
        tab.set("freq.start", None)
        tab.set("freq.stop", None)
    elif not isinstance(inputs.center_mhz, Keep) and not isinstance(
        inputs.span_mhz, Keep
    ):
        tab.set("freq.start", inputs.center_mhz - inputs.span_mhz / 2)
        tab.set("freq.stop", inputs.center_mhz + inputs.span_mhz / 2)
    tab.set("freq.n", inputs.points)
    tab.set("readout.gain", inputs.gain)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=OnetoneSpectrumSummary,
            updates={
                "model_type": inputs.model_type,
                "fit_bg_amp_slope": inputs.fit_bg_amp_slope,
                "fit_bg_phase_curvature": inputs.fit_bg_phase_curvature,
                "edelay": inputs.edelay,
            },
        )
        if analysis.result is None:
            return OnetoneSpectrumReport(
                frequency_mhz=None,
                linewidth_mhz=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return OnetoneSpectrumReport(
            fit_quality=analysis.result.fit_quality,
            frequency_mhz=report_number(analysis.result.freq),
            linewidth_mhz=report_number(analysis.result.fwhm),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = onetone_spectrum
