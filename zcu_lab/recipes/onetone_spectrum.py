"""Resonator frequency sweep, spectral fit and reviewed calibration."""

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_contracts import RecipeNeedsParameters

from .helpers import (
    FrequencySweepCalibration,
    FrequencySweepRange,
    report_number,
    retain_scalar,
    set_frequency_sweep,
)


class OnetoneSpectrumInputs(FormModel):
    """Inputs for onetone spectrum; all units and null policies are declared per
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
    center_mhz: float | None = Field(
        None, description="Sweep center in MHz. Null reads $resonator.freq."
    )
    span_mhz: float | None = Field(
        None,
        description="Full frequency span in MHz; endpoints are center minus/plus span/2. Null uses 3 times $resonator.kappa as the full span.",
    )
    gain: float | None = Field(
        None,
        description="Fixed pulse amplitude in native gain units; null keeps the GUI value.",
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


class OnetoneSpectrumReport(FormModel):
    """Onetone spectrum results and failure reason; cancellation returns no report."""

    frequency_mhz: float | None = Field(
        description="Fitted resonator frequency in MHz; null means not estimable."
    )
    linewidth_mhz: float | None = Field(
        description="Fitted full linewidth in MHz; null means not estimable."
    )
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class OnetoneSpectrumSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    freq: float | None = Field(
        description="Owner freq in MHz; null retains an unavailable estimate."
    )

    fwhm: float | None = Field(
        description="Owner fwhm in MHz; null retains an unavailable estimate."
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

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Expected analysis failures return a reason in
    the report; other errors propagate. Save raw data and completed artifacts once
    in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    for name, value in (
        ("reuse_tab_id", inputs.reuse_tab_id),
        ("readout_ref", inputs.readout_ref),
    ):
        if value is not None and (not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    if inputs.span_mhz is not None and inputs.span_mhz <= 0:
        raise ValueError("span_mhz must be positive")
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "onetone/freq",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    missing = set_frequency_sweep(
        tab,
        ("sweep", "freq"),
        calibration=FrequencySweepCalibration(
            "$resonator.freq", "$resonator.kappa", 3.0
        ),
        session=session,
        scan=FrequencySweepRange(
            center_mhz=inputs.center_mhz, span_mhz=inputs.span_mhz, expts=inputs.points
        ),
    )
    if missing:
        raise RecipeNeedsParameters(missing)
    if inputs.gain is not None:
        tab.select(("modules", "readout"), inline=True)
        tab.set(("modules", "readout", "gain"), inputs.gain)
    else:
        retain_scalar(tab, ("modules", "readout", "gain"))
    tab.set(("reps",), inputs.reps)
    tab.set(("rounds",), inputs.rounds)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Analyze the acquisition with the GUI parameters.
        analysis = await tab.analyze("primary", result=OnetoneSpectrumSummary)
        if analysis.result is None:
            return OnetoneSpectrumReport(
                frequency_mhz=None,
                linewidth_mhz=None,
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
        return OnetoneSpectrumReport(
            frequency_mhz=report_number(analysis.result.freq),
            linewidth_mhz=report_number(analysis.result.fwhm),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = onetone_spectrum
