"""Resonator frequency/gain map with raw save and Run preview only."""

from pydantic import Field
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_contracts import RecipeNeedsParameters

from .helpers import (
    FrequencySweepCalibration,
    FrequencySweepRange,
    retain_scalar,
    set_frequency_sweep,
)


class OnetoneSpectrumOverPowerInputs(FormModel):
    """Inputs for onetone spectrum over power; all units and null policies are
    declared per field."""

    target: str | None = Field(
        None,
        description="Qubit alias for this Reset; null uses the GUI aliases without changing focus.",
    )
    reuse_tab_id: str | None = Field(
        None,
        description="Idle matching tab to reset; null opens a new tab. May be combined with target.",
    )
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


class OnetoneSpectrumOverPowerReport(FormModel):
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

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Owner, acquisition, save and analysis errors
    propagate. Save raw data and completed artifacts once in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    for name, value in (
        ("reuse_tab_id", inputs.reuse_tab_id),
        ("readout_ref", inputs.readout_ref),
    ):
        if value is not None and (not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    if inputs.span_mhz is not None and inputs.span_mhz <= 0:
        raise ValueError("span_mhz must be positive")
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "onetone/power_dep",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    missing = set_frequency_sweep(
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
    if missing:
        raise RecipeNeedsParameters(missing)
    if inputs.gain_range is not None and inputs.gain_points is not None:
        tab.set(
            ("sweep", "gain"),
            LinRange(
                start=inputs.gain_range[0],
                stop=inputs.gain_range[1],
                n=inputs.gain_points,
            ),
        )
    else:
        tab.set(
            ("sweep", "gain", "start"),
            None if inputs.gain_range is None else inputs.gain_range[0],
        )
        tab.set(
            ("sweep", "gain", "stop"),
            None if inputs.gain_range is None else inputs.gain_range[1],
        )
        tab.set(("sweep", "gain", "n"), inputs.gain_points)
    retain_scalar(tab, ("modules", "readout", "gain"))
    tab.set(("reps",), inputs.reps)
    tab.set(("rounds",), inputs.rounds)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        tab.preview()
        return OnetoneSpectrumOverPowerReport(map_acquired=True)
    finally:
        tab.save()


DEFINITION = onetone_spectrum_over_power
