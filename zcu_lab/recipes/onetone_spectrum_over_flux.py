"""Resonator frequency/flux map and interactive flux calibration.

Recipe freq_points → sweep.freq.expts; flux_points → sweep.flux.expts.
Both scan counts appear as Expts in the GUI.
"""

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from .helpers import (
    FrequencySweepCalibration,
    FrequencySweepRange,
    report_number,
    retain_scalar,
    set_flux_sweep,
    set_frequency_sweep,
    use_flux_device,
)


class OnetoneSpectrumOverFluxInputs(FormModel):
    """Inputs for onetone spectrum over flux; all units and null policies are
    declared per field."""

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
        description="Full frequency span in MHz; endpoints are center minus/plus span/2. Null uses 2 times $resonator.kappa as the full span.",
    )
    gain: float | None = Field(
        None,
        description="Fixed pulse amplitude in native gain units; null keeps the GUI value.",
    )
    flux_device: str = Field(
        description="Registered flux-device name; read only its cached snapshot, without connecting."
    )
    flux_unit: str = Field(
        description="Exact cached device unit, including native; no coordinate conversion occurs."
    )
    freq_points: int | None = Field(
        None, description="Frequency sample count; null keeps the GUI count."
    )
    flux_points: int = Field(
        description="Required native-flux sample count, at least two."
    )
    flux_range: list[float] = Field(
        description="Two distinct device-native flux endpoints, in either direction; no conversion occurs."
    )
    reps: int | None = Field(
        None, description="Acquisition repetition count; null keeps the GUI value."
    )
    rounds: int | None = Field(
        None, description="Averaging round count; null keeps the GUI value."
    )


class OnetoneSpectrumOverFluxReport(FormModel):
    """Onetone spectrum over flux results and failure reason; cancellation returns no report."""

    flux_half: float | None = Field(
        description="Accepted half-flux line in device-native coordinates; null means not estimable."
    )
    flux_integer: float | None = Field(
        description="Accepted integer-flux line in device-native coordinates; null means not estimable."
    )
    flux_period: float | None = Field(
        description="Flux period in device-native coordinates; null means not estimable."
    )
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class FluxPickSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    flx_half: float | None = Field(
        description="Owner flx_half in native flux coordinate; null retains an unavailable estimate."
    )

    flx_int: float | None = Field(
        description="Owner flx_int in native flux coordinate; null retains an unavailable estimate."
    )

    flx_period: float | None = Field(
        description="Owner flx_period in native flux coordinate; null retains an unavailable estimate."
    )


@recipe(
    "onetone_spectrum_over_flux",
    description="Run one frequency/flux survey with explicit native flux range/count. Save raw and Primary analysis. flux_unit must match the cached device snapshot.",
)
async def onetone_spectrum_over_flux(
    session: RecipeSession, inputs: OnetoneSpectrumOverFluxInputs
) -> OnetoneSpectrumOverFluxReport:
    """Map the resonator spectrum across a native flux range.

    At each flux value, sweep the readout frequency and average IQ. Primary lets
    the user pick half- and integer-flux lines to estimate flux calibration.
    Ask before applying it, then save the raw map and analysis.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Expected analysis failures return a reason in
    the report; other errors propagate. Save raw data and completed artifacts once
    in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    if (
        not inputs.flux_device
        or not inputs.flux_device.strip()
        or (not inputs.flux_unit)
        or (not inputs.flux_unit.strip())
    ):
        raise ValueError("flux_device and flux_unit must be non-empty strings")
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
        "onetone/flux_dep",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    missing: list[MissingParameter] = []
    missing.extend(
        set_frequency_sweep(
            tab,
            ("sweep", "freq"),
            calibration=FrequencySweepCalibration(
                "$resonator.freq", "$resonator.kappa", 2.0
            ),
            scan=FrequencySweepRange(
                center_mhz=inputs.center_mhz,
                span_mhz=inputs.span_mhz,
                expts=inputs.freq_points,
            ),
        )
    )
    use_flux_device(tab, inputs.flux_device, unit=inputs.flux_unit)
    set_flux_sweep(
        tab,
        ("sweep", "flux"),
        start=inputs.flux_range[0],
        stop=inputs.flux_range[1],
        expts=inputs.flux_points,
    )
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
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
        analysis = await tab.analyze(
            "primary", result=FluxPickSummary, interactive=True
        )
        if analysis.result is None:
            return OnetoneSpectrumOverFluxReport(
                flux_half=None,
                flux_integer=None,
                flux_period=None,
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
        return OnetoneSpectrumOverFluxReport(
            flux_half=report_number(analysis.result.flx_half),
            flux_integer=report_number(analysis.result.flx_int),
            flux_period=report_number(analysis.result.flx_period),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = onetone_spectrum_over_flux
