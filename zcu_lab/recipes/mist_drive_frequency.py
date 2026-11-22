"""MIST probe frequency/gain survey with native save and map analysis."""

from typing import Annotated

from pydantic import Field
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


class MistDriveFrequencyInputs(RecipeInputs):
    """Set the frequency and gain rectangle for a resonator probe map."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    freq_range: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Set Probe carrier frequency sweep › Start and Stop (MHz), in that order.",
    )
    gain_range: list[float] = Field(
        min_length=2,
        max_length=2,
        description="Set Probe gain sweep › Start and Stop (a.u.), in that order.",
    )
    points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP,
        description="Set Expts (count) in Probe carrier frequency sweep and Probe gain sweep. At least 2.",
    )


class MistDriveFrequencyReport(RecipeReport):
    """MIST drive frequency results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default baseline-subtracted response map completed."
    )

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_drive_frequency",
    description="Map mean IQ shift versus resonator probe carrier and gain. Save the acquired complex IQ and Analyze map without writeback.",
)
async def mist_drive_frequency(
    session: RecipeSession, inputs: MistDriveFrequencyInputs
) -> MistDriveFrequencyReport:
    """Probe the requested MHz/gain rectangle and save the MIST response map.

    points → expts sets the sweep sample count.
    Analysis shows the IQ shift map and has no display inputs.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "mist/power_dep/drive_freq",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.set("freq.start", inputs.freq_range[0])
    tab.set("freq.stop", inputs.freq_range[1])
    tab.set("freq.n", inputs.points)
    tab.set("gain.start", inputs.gain_range[0])
    tab.set("gain.stop", inputs.gain_range[1])
    tab.set("gain.n", inputs.points)
    tab.apply_overrides(inputs.overrides)
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
