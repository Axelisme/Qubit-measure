"""One native flux/gain MIST survey, separate from the adaptive workflow."""

from typing import Annotated

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from zcu_lab.v2.mist.flux_dep.gui import MistFluxDepAnalysis


class MistFluxDepInputs(RecipeInputs):
    """Inputs for mist flux dep; all units and null policies are declared per
    field."""

    flux_device: str = Field(
        description="Registered flux-device name; read only its cached snapshot, without connecting."
    )
    flux_unit: str = Field(
        description="Exact cached device unit, including native; no coordinate conversion occurs."
    )
    flux_range: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Two distinct device-native flux endpoints, in either direction; no conversion occurs.",
    )
    gain_range: list[Annotated[float, Field(ge=-1, le=1)]] = Field(
        min_length=2,
        max_length=2,
        description="Two finite distinct native gain endpoints in [-1, 1].",
    )
    points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP, description="Sample count on each map axis, at least two."
    )

    flux_half: float | None | Keep = Field(
        KEEP,
        description="Half-flux device value; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    flux_period: float | None | Keep = Field(
        KEEP,
        description="Flux period (device units); omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    ac_coeff: float | None | Keep = Field(
        KEEP,
        description="Photon coefficient; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    secondary_xaxis: bool | None | Keep = Field(
        KEEP,
        description="Secondary device-value axis; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    auto_range: bool | None | Keep = Field(
        KEEP,
        description="Full acquired range; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class MistFluxDepReport(RecipeReport):
    """MIST flux dependence results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default raw-coordinate baseline-difference map completed."
    )

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_flux_dep",
    description="Acquire one MIST flux/gain map with explicit native ranges; save raw and Primary without writeback.",
)
async def mist_flux_dep(
    session: RecipeSession, inputs: MistFluxDepInputs
) -> MistFluxDepReport:
    """Acquire and save one MIST map using the current global aliases.

    points → expts sets the sweep sample count.
    Analysis controls: flux_half, flux_period, ac_coeff, secondary_xaxis, auto_range. Omission retains each analysis value; null is passed to its owner.
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
    tab.set("sweep.flux.start", inputs.flux_range[0])
    tab.set("sweep.flux.stop", inputs.flux_range[1])
    tab.set("sweep.flux.n", inputs.points)
    if inputs.gain_range[0] == inputs.gain_range[1]:
        raise ValueError("gain endpoints must be distinct")
    tab.set("sweep.gain.start", inputs.gain_range[0])
    tab.set("sweep.gain.stop", inputs.gain_range[1])
    tab.set("sweep.gain.n", inputs.points)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=MistFluxDepAnalysis,
            updates={
                key: value
                for key, value in {
                    "flux_half": inputs.flux_half,
                    "flux_period": inputs.flux_period,
                    "ac_coeff": inputs.ac_coeff,
                    "secondary_xaxis": inputs.secondary_xaxis,
                    "auto_range": inputs.auto_range,
                }.items()
                if not isinstance(value, Keep)
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
