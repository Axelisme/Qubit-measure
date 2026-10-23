"""Resonator frequency/gain map with raw save and Run preview only.

Recipe freq_points → sweep.freq.expts; gain_points → sweep.gain.expts.
Both scan counts appear as Expts in the GUI.
"""

from pydantic import Field
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)
from zcu_tools.mcp.measure.recipe_contracts import RecipeNeedsParameters

from .helpers import (
    FrequencySweepCalibration,
    FrequencySweepRange,
    frequency_sweep_edits,
)


class OnetoneSpectrumOverPowerInputs(RecipeInputs):
    """Inputs for onetone spectrum over power; all units and null policies are
    declared per field."""

    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the GUI selection. Frequency uses a valid captured library value, then resonator calibration.",
    )
    center_mhz: float | None = Field(
        None, description="Sweep center in MHz. Null reads $resonator.freq."
    )
    span_mhz: float | None = Field(
        None,
        description="Full frequency span in MHz; endpoints are center minus/plus span/2. Null uses 3 times $resonator.kappa as the full span.",
    )
    freq_points: int | None = Field(
        None, description="Frequency sample count; null keeps the GUI count."
    )
    gain_points: int | None = Field(
        None, description="Gain sample count; null keeps the GUI count."
    )
    gain_range: list[float] | None = Field(
        None, description="Two native gain endpoints; null keeps the GUI range."
    )
    reps: int | None = Field(
        None, description="Acquisition repetition count; null keeps the GUI value."
    )
    rounds: int | None = Field(
        None, description="Averaging round count; null keeps the GUI value."
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
    for name, value in (
        ("reuse_tab_id", inputs.reuse_tab_id),
        ("readout_ref", inputs.readout_ref),
    ):
        if value is not None and (not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    if inputs.span_mhz is not None and inputs.span_mhz <= 0:
        raise ValueError("span_mhz must be positive")
    tab = session.open_recipe_tab("onetone/power_dep", inputs, target_alias="qubit")
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select_sources(inputs)

    edits: list[FieldEdit] = []
    sweep = frequency_sweep_edits(
        tab,
        ("sweep", "freq"),
        calibration=FrequencySweepCalibration(
            "$resonator.freq", "$resonator.kappa", 3.0
        ),
        scan=FrequencySweepRange(
            center_mhz=inputs.center_mhz,
            span_mhz=inputs.span_mhz,
            expts=inputs.freq_points,
        ),
    )
    missing = sweep.missing
    edits.extend(sweep.edits)
    if missing:
        raise RecipeNeedsParameters(missing)
    if inputs.gain_range is not None and inputs.gain_points is not None:
        edits.append(
            FieldEdit(
                path=("sweep", "gain"),
                value=LinRange(
                    start=inputs.gain_range[0],
                    stop=inputs.gain_range[1],
                    n=inputs.gain_points,
                ),
            )
        )
    else:
        edits.append(
            FieldEdit(
                path=("sweep", "gain", "start"),
                value=None if inputs.gain_range is None else inputs.gain_range[0],
            )
        )
        edits.append(
            FieldEdit(
                path=("sweep", "gain", "stop"),
                value=None if inputs.gain_range is None else inputs.gain_range[1],
            )
        )
        edits.append(FieldEdit(path=("sweep", "gain", "n"), value=inputs.gain_points))
    tab.apply_inputs(
        inputs, {("reps",): inputs.reps, ("rounds",): inputs.rounds}, edits=edits
    )

    await tab.run()
    try:
        tab.preview()
        return OnetoneSpectrumOverPowerReport(map_acquired=True)
    finally:
        tab.save()


DEFINITION = onetone_spectrum_over_power
