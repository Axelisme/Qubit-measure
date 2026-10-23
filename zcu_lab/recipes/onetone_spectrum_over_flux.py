"""Resonator frequency/flux map and interactive flux calibration.

Recipe freq_points → sweep.freq.expts; flux_points → sweep.flux.expts.
Both scan counts appear as Expts in the GUI.
"""

from pydantic import Field
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
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

from .helpers import (
    FrequencySweepCalibration,
    FrequencySweepRange,
    flux_device_edits,
    flux_sweep_edits,
    frequency_sweep_edits,
)


class OnetoneSpectrumOverFluxInputs(RecipeInputs):
    """Inputs for onetone spectrum over flux; all units and null policies are
    declared per field."""

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


class OnetoneSpectrumOverFluxReport(RecipeReport):
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

    freq_points → sweep.freq.expts; flux_points → sweep.flux.expts.
    Analysis uses the captured GUI parameters; no fit controls are added.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
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
    tab = session.open_recipe_tab("onetone/flux_dep", inputs, target_alias="qubit")
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select_sources(inputs)
    edits: list[FieldEdit] = []
    missing: list[MissingParameter] = []
    sweep = frequency_sweep_edits(
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
    missing.extend(sweep.missing)
    edits.extend(sweep.edits)
    edits.extend(flux_device_edits(tab, inputs.flux_device, unit=inputs.flux_unit))
    edits.extend(
        flux_sweep_edits(
            ("sweep", "flux"),
            start=inputs.flux_range[0],
            stop=inputs.flux_range[1],
            expts=inputs.flux_points,
        )
    )
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    if inputs.gain is not None:
        tab.select(("modules", "readout"), inline=True)
        edits.append(FieldEdit(path=("modules", "readout", "gain"), value=inputs.gain))
    tab.apply_inputs(
        inputs, {("reps",): inputs.reps, ("rounds",): inputs.rounds}, edits=edits
    )
    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=FluxPickResult, interactive=True)
        if analysis.result is None:
            return OnetoneSpectrumOverFluxReport(
                flux_half=None,
                flux_integer=None,
                flux_period=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )
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
