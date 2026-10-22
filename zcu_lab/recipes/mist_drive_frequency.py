"""MIST probe frequency/gain survey with native save and map analysis."""

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe


class MistDriveFrequencyInputs(FormModel):
    """Inputs for mist drive frequency; all units and null policies are declared
    per field."""

    target: str | None = Field(
        None,
        description="Qubit alias for this Reset; null uses the GUI aliases without changing focus.",
    )
    freq_range: list[float] = Field(description="Two frequency endpoints in MHz.")
    gain_range: list[float] = Field(
        description="Two native gain endpoints; null keeps the GUI range."
    )
    points: int = Field(51, description="Sample count on each map axis, at least two.")


class MistDriveFrequencyReport(FormModel):
    """MIST drive frequency results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default baseline-subtracted response map completed."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class MistDriveFrequencySummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_drive_frequency",
    description="Map MIST probe frequency/gain, save native IQ and analyze the response. No writeback.",
)
async def mist_drive_frequency(
    session: RecipeSession, inputs: MistDriveFrequencyInputs
) -> MistDriveFrequencyReport:
    """Probe the requested MHz/gain rectangle and save the MIST response map.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Expected analysis failures return a reason in
    the report; other errors propagate. Save raw data and completed artifacts once
    in the Run finalizer.
    """
    if len(inputs.freq_range) != 2 or len(inputs.gain_range) != 2:
        raise ValueError("freq_range and gain_range must each contain two endpoints")
    if inputs.points < 2:
        raise ValueError("points must be at least two")
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "mist/power_dep/drive_freq",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
    )
    tab.set(
        ("sweep", "freq"),
        LinRange(
            start=inputs.freq_range[0], stop=inputs.freq_range[1], n=inputs.points
        ),
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
        analysis = await tab.analyze("primary", result=MistDriveFrequencySummary)
        if analysis.result is None:
            return MistDriveFrequencyReport(
                analysis_completed=False,
                analysis_failure=analysis.reason,
            )
        tab.preview()
        return MistDriveFrequencyReport(analysis_completed=True)
    finally:
        tab.save()


DEFINITION = mist_drive_frequency
