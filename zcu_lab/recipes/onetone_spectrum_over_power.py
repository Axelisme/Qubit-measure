"""Resonator frequency/gain map with raw save and Run preview only.

Recipe freq_points → freq.expts; gain_points → gain.expts.
Both scan counts appear as Expts in the GUI.
"""

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
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)


class OnetoneSpectrumOverPowerInputs(RecipeInputs):
    """Inputs for onetone spectrum over power; all units and null policies are
    declared per field."""

    target: Annotated[
        SingleTarget,
        Field(
            description="Qubit binding used to resolve its resonator; omit or pass null to keep the current alias."
        ),
    ]
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Select the Readout pulse and ADC window by library key; omit to retain the Form selection.",
    )
    center_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Readout RF frequency › Start and Stop around this RF center (MHz); pair with span_mhz.",
    )
    span_mhz: Annotated[float, Field(gt=0)] | None | Keep = Field(
        KEEP,
        description="Set Readout RF frequency › Start and Stop (MHz) to center_mhz minus/plus half this full span.",
    )
    freq_points: Annotated[int, Field(ge=2)] | None | Keep = Field(
        KEEP,
        description="Set Readout RF frequency › Expts (count), the number of frequencies at each outer sweep setting.",
    )
    gain_points: int | None | Keep = Field(
        KEEP,
        description="Set Readout gain › Expts (count), the number of readout drive amplitudes.",
    )
    gain_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP,
        description="Set Readout gain › Start and Stop (a.u.), the two DAC full-scale fractions.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), shots per frequency per round.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), sweeps to average at each outer setting.",
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

    freq_points → freq.expts; gain_points → gain.expts.
    Omitted inputs retain the Form values and null passes to the Form.
    Missing Form values stop before acquisition.
    """
    tab = session.open_tab(
        "onetone/power_dep",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
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
    if inputs.gain_range is None:
        tab.set("gain.start", None)
        tab.set("gain.stop", None)
    elif not isinstance(inputs.gain_range, Keep):
        tab.set("gain.start", inputs.gain_range[0])
        tab.set("gain.stop", inputs.gain_range[1])
    tab.set("gain.n", inputs.gain_points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        tab.preview()
        return OnetoneSpectrumOverPowerReport(map_acquired=True)
    finally:
        tab.save()


DEFINITION = onetone_spectrum_over_power
