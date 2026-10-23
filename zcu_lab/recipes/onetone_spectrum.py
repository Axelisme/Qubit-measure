"""Resonator frequency sweep, spectral fit and reviewed calibration.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Annotated, Literal

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
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2.onetone.freq.gui import OnetoneSpectrumSummary


class OnetoneSpectrumInputs(RecipeInputs):
    """Inputs for onetone spectrum; all units and null policies are declared per
    field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value; null clears/disables it. Sources are chosen by the Form.",
    )
    center_mhz: float | None | Keep = Field(
        KEEP,
        description="Sweep center in MHz. Supply with span_mhz; omission retains the Form range, null clears it.",
    )
    span_mhz: Annotated[float, Field(gt=0)] | None | Keep = Field(
        KEEP,
        description="Full frequency span in MHz; endpoints are center minus/plus span/2. Supply with center_mhz; omission retains the Form range, null clears it.",
    )
    gain: float | None | Keep = Field(
        KEEP,
        description="Fixed pulse amplitude in native gain units; omission retains the Form value; null clears/disables it.",
    )
    points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP,
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

    model_type: Literal["hm", "t", "auto"] | None | Keep = Field(
        KEEP,
        description="Model type; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    fit_bg_amp_slope: bool | None | Keep = Field(
        KEEP,
        description="Fit amplitude background; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    fit_bg_phase_curvature: bool | None | Keep = Field(
        KEEP,
        description="Fit phase curvature; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    edelay: float | None | Keep = Field(
        KEEP,
        description="Electrical delay seed (us; blank = auto); omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class OnetoneSpectrumReport(RecipeReport):
    """Onetone spectrum results and failure reason; cancellation returns no report."""

    frequency_mhz: float | None = Field(
        description="Fitted resonator frequency in MHz; null means not estimable."
    )
    linewidth_mhz: float | None = Field(
        description="Fitted full linewidth in MHz; null means not estimable."
    )


@recipe(
    "onetone_spectrum",
    description="Run one onetone spectrum and save raw data and Primary analysis.",
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
    tab.select("modules.readout", inputs.readout_ref)
    if not isinstance(inputs.gain, Keep):
        pulse_selector = tab.read(("modules", "readout")).selector
        if pulse_selector is not None and pulse_selector.mode == "ref":
            tab.command("modules.readout", "to_inline")
    if isinstance(inputs.center_mhz, Keep) != isinstance(inputs.span_mhz, Keep):
        missing = "span_mhz" if isinstance(inputs.span_mhz, Keep) else "center_mhz"
        raise RecipeNeedsParameters(
            (MissingParameter(missing, "Supply center_mhz and span_mhz together"),)
        )
    if inputs.center_mhz is None or inputs.span_mhz is None:
        tab.set("sweep.freq.start", None)
        tab.set("sweep.freq.stop", None)
    elif not isinstance(inputs.center_mhz, Keep) and not isinstance(
        inputs.span_mhz, Keep
    ):
        tab.set("sweep.freq.start", inputs.center_mhz - inputs.span_mhz / 2)
        tab.set("sweep.freq.stop", inputs.center_mhz + inputs.span_mhz / 2)
    tab.set("sweep.freq.n", inputs.points)
    tab.set("modules.readout.gain", inputs.gain)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=OnetoneSpectrumSummary,
            updates={
                key: value
                for key, value in {
                    "model_type": inputs.model_type,
                    "fit_bg_amp_slope": inputs.fit_bg_amp_slope,
                    "fit_bg_phase_curvature": inputs.fit_bg_phase_curvature,
                    "edelay": inputs.edelay,
                }.items()
                if not isinstance(value, Keep)
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
            frequency_mhz=report_number(analysis.result.freq),
            linewidth_mhz=report_number(analysis.result.fwhm),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = onetone_spectrum
