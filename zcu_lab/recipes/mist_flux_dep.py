"""One native flux/gain MIST survey, separate from the adaptive workflow."""

from math import isfinite

from pydantic import Field
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)

from zcu_lab.v2.mist.flux_dep.gui import MistFluxDepAnalysis

from .helpers import (
    flux_device_edits,
    flux_sweep_edits,
)


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
        description="Two distinct device-native flux endpoints, in either direction; no conversion occurs."
    )
    gain_range: list[float] = Field(
        description="Two finite distinct native gain endpoints in [-1, 1]."
    )
    points: int = Field(101, description="Sample count on each map axis, at least two.")

    flux_half: float | None = Field(
        None,
        description="Half-flux device value; null keeps the GUI analysis value. Analysis only.",
    )
    flux_period: float | None = Field(
        None,
        description="Flux period (device units); null keeps the GUI analysis value. Analysis only.",
    )
    ac_coeff: float | None = Field(
        None,
        description="Photon coefficient; null keeps the GUI analysis value. Analysis only.",
    )
    secondary_xaxis: bool | None = Field(
        None,
        description="Secondary device-value axis; null keeps the GUI analysis value. Analysis only.",
    )
    auto_range: bool | None = Field(
        None,
        description="Full acquired range; null keeps the GUI analysis value. Analysis only.",
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
    Analysis controls: flux_half, flux_period, ac_coeff, secondary_xaxis, auto_range. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    for name, values in (
        ("flux_range", inputs.flux_range),
        ("gain_range", inputs.gain_range),
    ):
        if (
            len(values) != 2
            or any((isinstance(value, bool) or not isfinite(value) for value in values))
            or values[0] == values[1]
        ):
            raise ValueError(f"{name} requires two finite distinct endpoints")
    if any((abs(value) > 1 for value in inputs.gain_range)):
        raise ValueError("gain_range endpoints must lie in [-1, 1]")
    if isinstance(inputs.points, bool) or inputs.points < 2:
        raise ValueError("points must be an integer >=2")
    tab = session.open_recipe_tab("mist/flux_dep", inputs, target_alias="qubit")
    tab.select_sources(inputs)

    edits: list[FieldEdit] = []
    edits.extend(flux_device_edits(tab, inputs.flux_device, unit=inputs.flux_unit))
    edits.extend(
        flux_sweep_edits(
            ("sweep", "flux"),
            start=inputs.flux_range[0],
            stop=inputs.flux_range[1],
            expts=inputs.points,
        )
    )
    tab.apply_inputs(
        inputs,
        {
            ("sweep", "gain"): LinRange(
                start=inputs.gain_range[0], stop=inputs.gain_range[1], n=inputs.points
            )
        },
        edits=edits,
    )

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
                if value is not None
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
