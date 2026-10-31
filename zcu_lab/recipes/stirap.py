"""Acquire a complete two-generator STIRAP scan with saved-readout analysis."""

from typing import Literal

from pydantic import Field
from zcu_tools.analysis.singleshot import Centroid3IQAnalysis
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)


class StirapInputs(RecipeInputs):
    """Operator-selected scan mode, carriers and controls; no tuning policy."""

    scan_mode: Literal["delay", "amplitude", "detuning"] = Field(
        description="Required mode: Stokes lead, common gain scale or Cartesian Pump/Stokes detuning; no auto choice."
    )
    start: float | None | Keep = Field(
        KEEP,
        description="Start: Stokes lead (us) for delay, common gain scale (unitless) for amplitude, Pump detuning (MHz).",
    )
    stop: float | None | Keep = Field(
        KEEP,
        description="First-axis stop, with the same mode-dependent units as start.",
    )
    n: int | None | Keep = Field(
        KEEP,
        description="First-axis sample count: Form n to cfg expts.",
    )
    stokes_start_mhz: float | None | Keep = Field(
        KEEP,
        description="Stokes detuning start in MHz, used only in detuning mode.",
    )
    stokes_stop_mhz: float | None | Keep = Field(
        KEEP,
        description="Stokes detuning stop in MHz, used only in detuning mode.",
    )
    stokes_n: int | None | Keep = Field(
        KEEP,
        description="Stokes-axis sample count, used only in detuning mode: Form n to cfg expts.",
    )
    pump_freq_mhz: float | None | Keep = Field(
        KEEP,
        description="Operator-measured 0-3 Pump carrier in MHz, never inferred from a model.",
    )
    stokes_freq_mhz: float | None | Keep = Field(
        KEEP,
        description="Operator-measured 3-2 Stokes carrier in MHz, never inferred from a model.",
    )
    pump_ch: int | None | Keep = Field(
        KEEP,
        description="Pump generator, separately bound from Stokes.",
    )
    stokes_ch: int | None | Keep = Field(
        KEEP,
        description="Stokes generator, separately bound from Pump.",
    )
    pump_gain: float | None | Keep = Field(
        KEEP,
        description="Signed Pump gain in native units.",
    )
    stokes_gain: float | None | Keep = Field(
        KEEP,
        description="Signed Stokes gain in native units.",
    )
    width_us: float | None | Keep = Field(
        KEEP,
        description="Common Gaussian sigma in us; each pulse lasts six sigma.",
    )
    stokes_lead_us: float | None | Keep = Field(
        KEEP,
        description="Fixed lead in us outside delay mode; positive means Stokes first.",
    )
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key with acquired centroid3 calibration.",
    )
    shots: int | None | Keep = Field(
        KEEP,
        description="Repetitions per scan point, classified into saved g/e/f counts.",
    )
    relax_delay_us: float | None | Keep = Field(
        KEEP,
        description="Relaxation interval in us for ground preparation.",
    )


class StirapReport(RecipeReport):
    """Saved-readout matrix quality; population arrays remain in analysis artifacts."""

    condition_number: float | None = Field(
        default=None,
        description="Saved training confusion matrix 2-norm condition number; null if analysis failed.",
    )
    minimum_singular_value: float | None = Field(
        default=None,
        description="Saved training confusion matrix minimum singular value; null if analysis failed.",
    )


@recipe(
    "stirap",
    description="Run an operator-selected STIRAP scan, analyze g/e/f counts and save counts and analysis.",
)
async def stirap(session: RecipeSession, inputs: StirapInputs) -> StirapReport:
    """Overlap Stokes/Pump Gaussians, read out after both, analyze and save.

    scan_mode is required; no automatic mode choice or tuning is performed.
    start/stop/n control Stokes lead in us (delay), dimensionless common gain
    scale (amplitude), or Pump detuning in MHz (detuning). Detuning mode also
    uses stokes_start_mhz/stokes_stop_mhz/stokes_n for the Cartesian second axis;
    those inputs are unused in other modes. n and stokes_n set Form n, lowered
    to cfg expts. Positive lead delays Pump; negative lead delays Stokes.
    Detunings add to their operator-measured carriers. Gains are signed native
    values; amplitude mode multiplies both by the common scale, not the width.
    Width is Gaussian sigma in us; each pulse lasts six sigma.

    Omission (Python KEEP, not a JSON value) retains each Form value/source.
    Explicit null goes unchanged to its owner. Ordered overrides follow named
    inputs, then the Form reports missing/invalid inputs before Run. Analysis
    uses only this run's saved readout centroid3 calibration, never live
    calibration or simulator truth. No writeback is proposed or requested.
    Analysis failure still saves g/e/f counts; owner errors propagate after Save.
    Save receipts identify failed outputs, not just completion of the recipe.
    """
    tab = session.open_tab(
        "twotone/stirap",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("scan", inputs.scan_mode)
    tab.set("pump_freq", inputs.pump_freq_mhz)
    tab.set("stokes_freq", inputs.stokes_freq_mhz)
    tab.set("pump_ch", inputs.pump_ch)
    tab.set("stokes_ch", inputs.stokes_ch)
    tab.set("pump_gain", inputs.pump_gain)
    tab.set("stokes_gain", inputs.stokes_gain)
    tab.set("width", inputs.width_us)
    tab.set("stokes_lead", inputs.stokes_lead_us)
    if inputs.scan_mode == "delay":
        tab.set("scan.stokes_lead.start", inputs.start)
        tab.set("scan.stokes_lead.stop", inputs.stop)
        tab.set("scan.stokes_lead.n", inputs.n)
    elif inputs.scan_mode == "amplitude":
        tab.set("scan.scale.start", inputs.start)
        tab.set("scan.scale.stop", inputs.stop)
        tab.set("scan.scale.n", inputs.n)
    else:
        tab.set("scan.pump.start", inputs.start)
        tab.set("scan.pump.stop", inputs.stop)
        tab.set("scan.pump.n", inputs.n)
        tab.set("scan.stokes.start", inputs.stokes_start_mhz)
        tab.set("scan.stokes.stop", inputs.stokes_stop_mhz)
        tab.set("scan.stokes.n", inputs.stokes_n)
    tab.set("shots", inputs.shots)
    tab.set("relax_delay", inputs.relax_delay_us)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=Centroid3IQAnalysis)
        if analysis.result is None:
            return StirapReport(analysis_failure=analysis.reason)
        return StirapReport(
            condition_number=report_number(analysis.result.condition_number),
            minimum_singular_value=report_number(
                analysis.result.minimum_singular_value
            ),
        )
    finally:
        tab.save()


DEFINITION = stirap
