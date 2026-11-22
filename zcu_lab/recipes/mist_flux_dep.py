"""One native flux/gain MIST survey, separate from the adaptive workflow."""

from typing import Annotated

from pydantic import Field
from zcu_tools.cfg_model import DeviceName
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
)
from zcu_tools.mcp.measure.session import GuiRpcError

from zcu_lab.v2.mist.flux_dep.gui import MistFluxDepAnalysis


class MistFluxDepInputs(RecipeInputs):
    """Set native bias and probe gain ranges and optional Analyze axes."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    flux_device: DeviceName = Field(
        description="Set Flux device to the connected DC bias source name. Check the unit from its captured settings."
    )
    flux_unit: str = Field(
        description="Set Flux-bias unit to the device unit (A, V or native); no conversion is applied."
    )
    flux_range: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Set Flux-bias sweep › Start and Stop in the selected Flux-bias unit, in either direction.",
    )
    gain_range: list[Annotated[float, Field(ge=-1, le=1)]] = Field(
        min_length=2,
        max_length=2,
        description="Set Probe gain sweep › Start and Stop (a.u.), in that order within [-1, 1].",
    )
    points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP,
        description="Set Expts (count) in Flux-bias sweep and Probe gain sweep. At least 2.",
    )

    flux_half: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Half-flux bias (device units). Null keeps native bias coordinates.",
    )
    flux_period: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Flux period (device units / Φ₀). Null keeps native bias coordinates.",
    )
    ac_coeff: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Photon number / gain squared. Omit to retain it; null keeps the gain axis in a.u.",
    )
    auto_range: bool | None | Keep = Field(
        KEEP,
        description="Set Analyze › Full acquired range. True uses endpoints; false uses automatic limits.",
    )


class MistFluxDepReport(RecipeReport):
    """MIST flux dependence results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the baseline-difference map completed."
    )

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_flux_dep",
    description="Map mean IQ shift versus native DC bias and resonator probe gain. Save the acquired complex IQ and Analyze map without writeback.",
)
async def mist_flux_dep(
    session: RecipeSession, inputs: MistFluxDepInputs
) -> MistFluxDepReport:
    """Acquire and save one MIST map using the current global aliases.

    points → expts sets the sweep sample count.
    Analysis controls: flux_half, flux_period, ac_coeff, auto_range. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "mist/flux_dep",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
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
    tab.set("flux.n", inputs.points)
    if inputs.gain_range[0] == inputs.gain_range[1]:
        raise ValueError("gain endpoints must be distinct")
    tab.set("gain.start", inputs.gain_range[0])
    tab.set("gain.stop", inputs.gain_range[1])
    tab.set("gain.n", inputs.points)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=MistFluxDepAnalysis,
            updates={
                "flux_half": inputs.flux_half,
                "flux_period": inputs.flux_period,
                "ac_coeff": inputs.ac_coeff,
                "auto_range": inputs.auto_range,
            },
        )
        if analysis.result is None:
            return MistFluxDepReport(
                analysis_completed=False, analysis_failure=analysis.reason
            )
        return MistFluxDepReport(analysis_completed=True)
    finally:
        tab.save()


DEFINITION = mist_flux_dep
