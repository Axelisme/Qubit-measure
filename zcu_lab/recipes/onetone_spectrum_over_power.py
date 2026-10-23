"""Resonator frequency/gain map with raw save and Run preview only.

Recipe freq_points → sweep.freq.expts; gain_points → sweep.gain.expts.
Both scan counts appear as Expts in the GUI.
"""

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
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)


class OnetoneSpectrumOverPowerInputs(RecipeInputs):
    """Inputs for onetone spectrum over power; all units and null policies are
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
    freq_points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP,
        description="Frequency sample count; omission retains the Form value; null clears/disables it.",
    )
    gain_points: int | None | Keep = Field(
        KEEP,
        description="Gain sample count; omission retains the Form value; null clears/disables it.",
    )
    gain_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP,
        description="Two native gain endpoints; omission retains the Form value; null clears/disables it.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Acquisition repetition count; omission retains the Form value; null clears/disables it.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count; omission retains the Form value; null clears/disables it.",
    )


class OnetoneSpectrumOverPowerReport(RecipeReport):
    """Completed onetone spectrum over power results; cancellation returns no
    report."""

    map_acquired: bool = Field(
        description="Whether the frequency/gain map completed and reached raw save; this is not a save receipt."
    )


@recipe(
    "onetone_spectrum_over_power",
    description="Run one frequency/gain survey, save raw and return its Run preview. No analysis or writeback.",
)
async def onetone_spectrum_over_power(
    session: RecipeSession, inputs: OnetoneSpectrumOverPowerInputs
) -> OnetoneSpectrumOverPowerReport:
    """Map the resonator response versus readout frequency and gain.

    At each gain, sweep readout frequency and average IQ to show how the
    resonance changes with drive amplitude. Save the raw map and capture its Run
    preview only; this survey performs no analysis or writeback.

    freq_points → sweep.freq.expts; gain_points → sweep.gain.expts.
    Analysis uses the captured GUI parameters; no fit controls are added.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "onetone/power_dep",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("modules.readout", inputs.readout_ref)
    if isinstance(inputs.center_mhz, Keep) != isinstance(inputs.span_mhz, Keep):
        missing = "span_mhz" if isinstance(inputs.span_mhz, Keep) else "center_mhz"
        raise RecipeNeedsParameters(
            (MissingParameter(missing, "Supply center_mhz and span_mhz together"),)
        )
    if inputs.center_mhz is None or inputs.span_mhz is None:
        tab.set("sweep.freq.start", None)
        tab.set("sweep.freq.stop", None)
    elif not isinstance(inputs.center_mhz, Keep) and not isinstance(
        inputs.span_mhz, Keep
    ):
        tab.set("sweep.freq.start", inputs.center_mhz - inputs.span_mhz / 2)
        tab.set("sweep.freq.stop", inputs.center_mhz + inputs.span_mhz / 2)
    tab.set("sweep.freq.n", inputs.freq_points)
    if inputs.gain_range is None:
        tab.set("sweep.gain.start", None)
        tab.set("sweep.gain.stop", None)
    elif not isinstance(inputs.gain_range, Keep):
        tab.set("sweep.gain.start", inputs.gain_range[0])
        tab.set("sweep.gain.stop", inputs.gain_range[1])
    tab.set("sweep.gain.n", inputs.gain_points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        tab.preview()
        return OnetoneSpectrumOverPowerReport(map_acquired=True)
    finally:
        tab.save()


DEFINITION = onetone_spectrum_over_power
