"""Resonator frequency sweep, spectral fit and reviewed calibration.

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
from zcu_tools.mcp.measure.recipe_contracts import RecipeNeedsParameters

from zcu_lab.v2.onetone.freq.gui import OnetoneSpectrumSummary

from .helpers import (
    FrequencySweepCalibration,
    FrequencySweepRange,
    frequency_sweep_edits,
)


class OnetoneSpectrumInputs(RecipeInputs):
    """Inputs for onetone spectrum; all units and null policies are declared per
    field."""

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

    model_type: Literal["hm", "t", "auto"] | None = Field(
        None,
        description="Model type; null keeps the GUI analysis value. Analysis only.",
    )
    fit_bg_amp_slope: bool | None = Field(
        None,
        description="Fit amplitude background; null keeps the GUI analysis value. Analysis only.",
    )
    fit_bg_phase_curvature: bool | None = Field(
        None,
        description="Fit phase curvature; null keeps the GUI analysis value. Analysis only.",
    )
    edelay: float | None = Field(
        None,
        description="Electrical delay seed (us; blank = auto); null keeps the GUI analysis value. Analysis only.",
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
    Analysis controls: model_type, fit_bg_amp_slope, fit_bg_phase_curvature, edelay. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    for name, value in (
        ("reuse_tab_id", inputs.reuse_tab_id),
        ("readout_ref", inputs.readout_ref),
    ):
        if value is not None and (not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    if inputs.span_mhz is not None and inputs.span_mhz <= 0:
        raise ValueError("span_mhz must be positive")
    tab = session.open_recipe_tab("onetone/freq", inputs, target_alias="qubit")
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select_sources(inputs)

    edits: list[FieldEdit] = []
    sweep = frequency_sweep_edits(
        tab,
        ("sweep", "freq"),
        calibration=FrequencySweepCalibration(
            "$resonator.freq", "$resonator.kappa", 3.0
        ),
        scan=FrequencySweepRange(
            center_mhz=inputs.center_mhz, span_mhz=inputs.span_mhz, expts=inputs.points
        ),
    )
    missing = sweep.missing
    edits.extend(sweep.edits)
    if missing:
        raise RecipeNeedsParameters(missing)
    if inputs.gain is not None:
        tab.select(("modules", "readout"), inline=True)
        edits.append(FieldEdit(path=("modules", "readout", "gain"), value=inputs.gain))
    tab.apply_inputs(
        inputs, {("reps",): inputs.reps, ("rounds",): inputs.rounds}, edits=edits
    )

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
                if value is not None
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
