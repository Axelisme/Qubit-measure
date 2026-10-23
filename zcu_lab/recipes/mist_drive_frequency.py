"""MIST probe frequency/gain survey with native save and map analysis."""

from pydantic import Field
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)

from zcu_lab.v2.mist.power_dep.drive_freq.gui import DriveFreqAnalyzeResult


class MistDriveFrequencyInputs(RecipeInputs):
    """Inputs for mist drive frequency; all units and null policies are declared
    per field."""

    freq_range: list[float] = Field(description="Two frequency endpoints in MHz.")
    gain_range: list[float] = Field(
        description="Two native gain endpoints; null keeps the GUI range."
    )
    points: int = Field(51, description="Sample count on each map axis, at least two.")


class MistDriveFrequencyReport(RecipeReport):
    """MIST drive frequency results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default baseline-subtracted response map completed."
    )

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_drive_frequency",
    description="Map MIST probe frequency/gain, save native IQ and analyze the response. No writeback.",
)
async def mist_drive_frequency(
    session: RecipeSession, inputs: MistDriveFrequencyInputs
) -> MistDriveFrequencyReport:
    """Probe the requested MHz/gain rectangle and save the MIST response map.

    points → expts sets the sweep sample count.
    Analysis uses the captured GUI parameters; no fit controls are added.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    if len(inputs.freq_range) != 2 or len(inputs.gain_range) != 2:
        raise ValueError("freq_range and gain_range must each contain two endpoints")
    if inputs.points < 2:
        raise ValueError("points must be at least two")
    tab = session.open_recipe_tab(
        "mist/power_dep/drive_freq", inputs, target_alias="qubit"
    )
    tab.select_sources(inputs)

    edits: list[FieldEdit] = []
    tab.apply_inputs(
        inputs,
        {
            ("sweep", "freq"): LinRange(
                start=inputs.freq_range[0], stop=inputs.freq_range[1], n=inputs.points
            ),
            ("sweep", "gain"): LinRange(
                start=inputs.gain_range[0], stop=inputs.gain_range[1], n=inputs.points
            ),
        },
        edits=edits,
    )

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=DriveFreqAnalyzeResult)
        if analysis.result is None:
            return MistDriveFrequencyReport(
                analysis_completed=False, analysis_failure=analysis.reason
            )
        tab.preview()
        return MistDriveFrequencyReport(analysis_completed=True)
    finally:
        tab.save()


DEFINITION = mist_drive_frequency
