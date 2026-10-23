"""Qubit frequency/flux map and interactive flux calibration."""

from pydantic import Field
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from .helpers import (
    DeviceParams,
    DeviceReply,
)


class TwotoneFluxDepInputs(RecipeInputs):
    """Inputs for two-tone spectrum over flux; all units and null policies are
    declared per field."""

    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the captured GUI selection.",
    )
    center_mhz: float = Field(
        description="Required drive-frequency sweep center in MHz."
    )
    span_mhz: float = Field(
        gt=0,
        description="Full frequency span in MHz; endpoints are center minus/plus span/2. Required positive drive-frequency full span.",
    )
    gain: float | None = Field(
        None,
        description="Fixed qubit-drive pulse amplitude in native gain units; null keeps the GUI value.",
    )
    flux_device: str = Field(
        description="Registered flux-device name; read only its cached snapshot, without connecting."
    )
    flux_unit: str = Field(
        description="Exact cached device unit, including native; no coordinate conversion occurs."
    )
    freq_points: int = Field(
        ge=2,
        strict=True,
        description="Required drive-frequency sample count, at least two.",
    )
    flux_points: int = Field(
        ge=2,
        strict=True,
        description="Required native-flux sample count, at least two.",
    )
    flux_range: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Two distinct device-native flux endpoints, in either direction; no conversion occurs.",
    )
    reps: int | None = Field(
        None, description="Acquisition repetition count; null keeps the GUI value."
    )
    rounds: int | None = Field(
        None, description="Averaging round count; null keeps the GUI value."
    )


class TwotoneFluxDepReport(RecipeReport):
    """Two-tone spectrum over flux results and failure reason; cancellation returns no report."""

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
    "twotone_flux_dep",
    description="Run one frequency/flux survey with explicit native flux range/count. Save raw and Primary analysis. flux_unit must match the cached device snapshot.",
)
async def twotone_flux_dep(
    session: RecipeSession, inputs: TwotoneFluxDepInputs
) -> TwotoneFluxDepReport:
    """Map the qubit spectrum across a native flux range.

    At each flux value, sweep the qubit drive frequency and average IQ. Primary lets
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
    if inputs.flux_range[0] == inputs.flux_range[1]:
        raise ValueError("flux endpoints must be distinct")
    tab = session.open_recipe_tab("twotone/flux_dep", inputs, target_alias="qubit")
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select_sources(inputs)
    edits: list[FieldEdit] = []
    unit = tab.call(
        "device.snapshot", DeviceParams(name=inputs.flux_device), result=DeviceReply
    ).snapshot.unit
    if not unit or not unit.strip() or unit == "none" or (unit != inputs.flux_unit):
        raise GuiRpcError(
            "Flux unit must match a usable cached device coordinate",
            reason="invalid_device",
        )
    if inputs.gain is not None:
        tab.select(("modules", "qub_pulse"), inline=True)
    edits.extend(
        [
            FieldEdit(
                path=("sweep", "freq"),
                value=LinRange(
                    start=inputs.center_mhz - inputs.span_mhz / 2,
                    stop=inputs.center_mhz + inputs.span_mhz / 2,
                    n=inputs.freq_points,
                ),
            ),
            FieldEdit(
                path=("sweep", "flux"),
                value=LinRange(
                    start=inputs.flux_range[0],
                    stop=inputs.flux_range[1],
                    n=inputs.flux_points,
                ),
            ),
            FieldEdit(path=("flux_dev",), value=inputs.flux_device),
            FieldEdit(path=("flux_unit",), value=unit),
            FieldEdit(path=("modules", "qub_pulse", "gain"), value=inputs.gain),
            FieldEdit(path=("reps",), value=inputs.reps),
            FieldEdit(path=("rounds",), value=inputs.rounds),
        ]
    )
    tab.apply_inputs(inputs, {}, edits=edits)
    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=FluxPickResult, interactive=True)
        if analysis.result is None:
            return TwotoneFluxDepReport(
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
        return TwotoneFluxDepReport(
            flux_half=report_number(analysis.result.flx_half),
            flux_integer=report_number(analysis.result.flx_int),
            flux_period=report_number(analysis.result.flx_period),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = twotone_flux_dep
