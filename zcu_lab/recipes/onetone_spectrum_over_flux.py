"""Resonator frequency/flux map and interactive flux calibration.

Recipe freq_points → freq.expts; flux_points → flux.expts.
Both scan counts appear as Expts in the GUI.
"""

from typing import Annotated

from pydantic import Field
from zcu_tools.cfg_model import DeviceName
from zcu_tools.gui.app.measure.interactive_flux_pick import FluxPickResult
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
from zcu_tools.mcp.measure.session import GuiRpcError


class OnetoneSpectrumOverFluxInputs(RecipeInputs):
    """Inputs for onetone spectrum over flux; all units and null policies are
    declared per field."""

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
    flux_device: DeviceName = Field(
        description="Registered flux-device name; read only its cached snapshot, without connecting."
    )
    flux_unit: str = Field(
        description="Exact cached device unit, including native; no coordinate conversion occurs."
    )
    freq_points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP,
        description="Frequency sample count; omission retains the Form value; null clears/disables it.",
    )
    flux_points: int = Field(
        ge=2, description="Required native-flux sample count, at least two."
    )
    flux_range: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Two distinct device-native flux endpoints, in either direction; no conversion occurs.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Acquisition repetition count; omission retains the Form value; null clears/disables it.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count; omission retains the Form value; null clears/disables it.",
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

    freq_points → freq.expts; flux_points → flux.expts.
    Analysis uses the captured GUI parameters; no fit controls are added.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "onetone/flux_dep",
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
    tab.set("freq.n", inputs.freq_points)
    unit = session.device_unit(inputs.flux_device)
    if not unit.strip() or unit == "none" or unit != inputs.flux_unit:
        raise GuiRpcError(
            "Flux unit must match a usable cached device coordinate",
            reason="invalid_device",
        )
    tab.set("flux_dev", inputs.flux_device)
    tab.set("flux_unit", unit)
    if inputs.flux_range[0] == inputs.flux_range[1]:
        raise ValueError("flux endpoints must be distinct")
    tab.set("flux.start", inputs.flux_range[0])
    tab.set("flux.stop", inputs.flux_range[1])
    tab.set("flux.n", inputs.flux_points)
    tab.set("readout.gain", inputs.gain)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

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
