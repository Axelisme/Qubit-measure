"""MIST probe frequency/gain survey with native save and map analysis."""

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


class MistDriveFrequencyInputs(RecipeInputs):
    """Inputs for mist drive frequency; all units and null policies are declared
    per field."""

    freq_range: list[float] = Field(
        min_length=2, max_length=2, description="Two frequency endpoints in MHz."
    )
    gain_range: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Two native gain endpoints; omission retains the Form value; null clears/disables it.",
    )
    points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP, description="Sample count on each map axis, at least two."
    )


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
    tab = session.open_tab(
        "mist/power_dep/drive_freq",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.set("sweep.freq.start", inputs.freq_range[0])
    tab.set("sweep.freq.stop", inputs.freq_range[1])
    tab.set("sweep.freq.n", inputs.points)
    tab.set("sweep.gain.start", inputs.gain_range[0])
    tab.set("sweep.gain.stop", inputs.gain_range[1])
    tab.set("sweep.gain.n", inputs.points)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=dict)
        if analysis.result is None:
            return MistDriveFrequencyReport(
                analysis_completed=False, analysis_failure=analysis.reason
            )
        tab.preview()
        return MistDriveFrequencyReport(analysis_completed=True)
    finally:
        tab.save()


DEFINITION = mist_drive_frequency
