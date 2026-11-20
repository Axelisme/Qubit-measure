"""Acquire STIRAP detuning with independent envelopes and saved-readout analysis."""

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
    report_number,
)

from zcu_lab.v2.twotone.stirap.detuning.core import StirapDetuningSummary


class StirapDetuningInputs(RecipeInputs):
    """Measured carriers, Cartesian MHz offsets and independent envelope references."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    pump_start_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Pump detuning start in MHz, added to the measured carrier.",
    )
    pump_stop_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Pump detuning stop in MHz.",
    )
    pump_n: int | None | Keep = Field(
        default=KEEP,
        description="Pump-axis sample count: Form n to cfg expts.",
    )
    stokes_start_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Stokes detuning start in MHz.",
    )
    stokes_stop_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Stokes detuning stop in MHz.",
    )
    stokes_n: int | None | Keep = Field(
        default=KEEP,
        description="Stokes-axis sample count: Form n to cfg expts.",
    )
    pump_freq_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Operator-measured 0-3 Pump carrier in MHz, never inferred from a model.",
    )
    stokes_freq_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Operator-measured 3-2 Stokes carrier in MHz, never inferred from a model.",
    )
    pump_ch: int | None | Keep = Field(
        default=KEEP,
        description="Pump generator, separately bound from Stokes.",
    )
    stokes_ch: int | None | Keep = Field(
        default=KEEP,
        description="Stokes generator, separately bound from Pump.",
    )
    pump_gain: float | None | Keep = Field(
        default=KEEP,
        description="Signed Pump gain in native units.",
    )
    stokes_gain: float | None | Keep = Field(
        default=KEEP,
        description="Signed Stokes gain in native units.",
    )
    pump_waveform_ref: str | None | Keep = Field(
        default=KEEP,
        description="Pump library key or Inline: gauss/cosine/drag/arb. Overrides set shape parameters.",
    )
    stokes_waveform_ref: str | None | Keep = Field(
        default=KEEP,
        description="Stokes library key or Inline: gauss/cosine/drag/arb, independent of Pump.",
    )
    stokes_lead_us: float | None | Keep = Field(
        default=KEEP,
        description="Fixed lead in us; positive means Stokes first.",
    )
    readout_ref: str | None | Keep = Field(
        default=KEEP,
        description="Readout library key with acquired centroid3 calibration.",
    )
    shots: int | None | Keep = Field(
        default=KEEP,
        description="Repetitions per scan point, classified into saved g/e/f counts.",
    )
    relax_delay_us: float | None | Keep = Field(
        default=KEEP,
        description="Relaxation interval in us for ground preparation.",
    )


class StirapDetuningReport(RecipeReport):
    """Saved readout diagnostics and the peak within the measured scan grid."""

    f_population_estimate: float | None = Field(
        default=None,
        description="Largest finite corrected f population (unitless, unclipped) on this grid, not fidelity or a global optimum. Null when unavailable.",
    )
    pump_detuning_mhz: float | None = Field(
        default=None,
        description="Pump offset from the measured carrier in MHz at the grid peak; null when unavailable.",
    )
    stokes_detuning_mhz: float | None = Field(
        default=None,
        description="Stokes offset from the measured carrier in MHz at the grid peak; null when unavailable.",
    )

    condition_number: float | None = Field(
        default=None,
        description="Saved training confusion matrix 2-norm condition number; null if analysis failed.",
    )
    minimum_singular_value: float | None = Field(
        default=None,
        description="Saved training confusion matrix minimum singular value; null if analysis failed.",
    )


@recipe(
    "stirap_detuning",
    description="Hardware-sweep Pump/Stokes detuning with independent envelopes, analyze g/e/f and save.",
)
async def stirap_detuning(
    session: RecipeSession, inputs: StirapDetuningInputs
) -> StirapDetuningReport:
    """Hardware-sweep Stokes detuning outside Pump, overlap drives, analyze and save.

    pump_start_mhz/pump_stop_mhz and stokes_start_mhz/stokes_stop_mhz are MHz
    offsets added to operator-measured carriers. pump_n and stokes_n set Form n,
    lowered to cfg expts. One program acquires the full grid with both hardware
    sweeps. Positive stokes_lead_us delays Pump; negative delays
    Stokes. Gains are signed. Each gauss/cosine/drag/arb waveform owns its length.
    Omitted waveform refs retain each current Form source, initially Gaussian
    sigma 0.05 us and length 0.3 us. Use a waveform library key or an exact Form
    option such as Inline: cosine, then set pump_waveform.length in us through
    overrides. Arbitrary envelopes use the selected asset's duration.

    Omission (Python KEEP, not a JSON value) retains each Form value/source.
    Explicit null goes unchanged to its owner. Ordered overrides follow named
    inputs, then the Form reports missing/invalid inputs before Run. Analysis
    uses only this run's saved readout centroid3 calibration, never live
    calibration or simulator truth. No writeback is proposed or requested.
    Analysis failure still saves g/e/f counts; owner errors propagate after Save.
    Save receipts identify failed outputs, not just completion of the recipe.
    """
    tab = session.open_tab(
        "twotone/stirap/detuning",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("pump_waveform", inputs.pump_waveform_ref)
    tab.select("stokes_waveform", inputs.stokes_waveform_ref)
    tab.set("pump_freq", inputs.pump_freq_mhz)
    tab.set("stokes_freq", inputs.stokes_freq_mhz)
    tab.set("pump_ch", inputs.pump_ch)
    tab.set("stokes_ch", inputs.stokes_ch)
    tab.set("pump_gain", inputs.pump_gain)
    tab.set("stokes_gain", inputs.stokes_gain)
    tab.set("stokes_lead", inputs.stokes_lead_us)
    tab.set("pump_detuning.start", inputs.pump_start_mhz)
    tab.set("pump_detuning.stop", inputs.pump_stop_mhz)
    tab.set("pump_detuning.n", inputs.pump_n)
    tab.set("stokes_detuning.start", inputs.stokes_start_mhz)
    tab.set("stokes_detuning.stop", inputs.stokes_stop_mhz)
    tab.set("stokes_detuning.n", inputs.stokes_n)
    tab.set("shots", inputs.shots)
    tab.set("relax_delay", inputs.relax_delay_us)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=StirapDetuningSummary)
        if analysis.result is None:
            return StirapDetuningReport(analysis_failure=analysis.reason)
        return StirapDetuningReport(
            f_population_estimate=analysis.result.f_population_estimate,
            pump_detuning_mhz=analysis.result.pump_detuning_mhz,
            stokes_detuning_mhz=analysis.result.stokes_detuning_mhz,
            condition_number=report_number(analysis.result.condition_number),
            minimum_singular_value=report_number(
                analysis.result.minimum_singular_value
            ),
        )
    finally:
        tab.save()


DEFINITION = stirap_detuning
