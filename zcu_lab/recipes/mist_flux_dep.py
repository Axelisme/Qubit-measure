"""One native flux/gain MIST survey, separate from the adaptive workflow."""

from math import isfinite

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe

from .helpers import set_flux_sweep, use_flux_device


class MistFluxDepInputs(FormModel):
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


class MistFluxDepReport(FormModel):
    """MIST flux dependence results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default raw-coordinate baseline-difference map completed."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class MistFluxDepSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_flux_dep",
    description="Acquire one MIST flux/gain map with explicit native ranges; save raw and Primary without writeback.",
)
async def mist_flux_dep(
    session: RecipeSession, inputs: MistFluxDepInputs
) -> MistFluxDepReport:
    """Acquire and save one MIST map using the current global aliases.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Expected analysis failures return a reason in
    the report; other errors propagate. Save raw data and completed artifacts once
    in the Run finalizer.
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
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab("mist/flux_dep")
    use_flux_device(tab, inputs.flux_device, unit=inputs.flux_unit)
    set_flux_sweep(
        tab,
        ("sweep", "flux"),
        start=inputs.flux_range[0],
        stop=inputs.flux_range[1],
        expts=inputs.points,
    )
    tab.set(
        ("sweep", "gain"),
        LinRange(
            start=inputs.gain_range[0], stop=inputs.gain_range[1], n=inputs.points
        ),
    )

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Analyze the acquisition with the GUI parameters.
        analysis = await tab.analyze("primary", result=MistFluxDepSummary)
        if analysis.result is None:
            return MistFluxDepReport(
                analysis_completed=False,
                analysis_failure=analysis.reason,
            )
        return MistFluxDepReport(analysis_completed=True)
    finally:
        tab.save()


DEFINITION = mist_flux_dep
