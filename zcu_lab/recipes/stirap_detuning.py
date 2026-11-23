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

from zcu_lab.v2.twotone.stirap.detuning.core import (
    StirapDetuningClassifiedPeak,
    StirapDetuningSummary,
)


class StirapDetuningInputs(RecipeInputs):
    """Measured carriers, Cartesian MHz offsets and independent envelope references."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    pump_start_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Set Pump detuning › Start (MHz).",
    )
    pump_stop_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Set Pump detuning › Stop (MHz).",
    )
    pump_n: int | None | Keep = Field(
        default=KEEP,
        description="Set Pump detuning › Expts (count).",
    )
    stokes_start_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Set Stokes detuning › Start (MHz).",
    )
    stokes_stop_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Set Stokes detuning › Stop (MHz).",
    )
    stokes_n: int | None | Keep = Field(
        default=KEEP,
        description="Set Stokes detuning › Expts (count).",
    )
    pump_freq_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Set Pump carrier 0–3 (MHz) to the measured transition frequency.",
    )
    stokes_freq_mhz: float | None | Keep = Field(
        default=KEEP,
        description="Set Stokes carrier 3–2 (MHz) to the measured transition frequency.",
    )
    pump_ch: int | None | Keep = Field(
        default=KEEP,
        description="Set Pump generator (channel).",
    )
    stokes_ch: int | None | Keep = Field(
        default=KEEP,
        description="Set Stokes generator (channel).",
    )
    pump_gain: float | None | Keep = Field(
        default=KEEP,
        description="Set Pump gain (a.u.).",
    )
    stokes_gain: float | None | Keep = Field(
        default=KEEP,
        description="Set Stokes gain (a.u.).",
    )
    pump_waveform_ref: str | None | Keep = Field(
        default=KEEP,
        description="Set Pump waveform to a library key or exact inline option.",
    )
    stokes_waveform_ref: str | None | Keep = Field(
        default=KEEP,
        description="Set Stokes waveform to a library key or exact inline option.",
    )
    stokes_lead_us: float | None | Keep = Field(
        default=KEEP,
        description="Set Fixed Stokes lead (µs); positive starts Stokes first.",
    )
    readout_ref: str | None | Keep = Field(
        default=KEEP,
        description="Set Readout to a module with saved centroid3 calibration.",
    )
    shots: int | None | Keep = Field(
        default=KEEP,
        description="Set Shots per point (count).",
    )
    relax_delay_us: float | None | Keep = Field(
        default=KEEP,
        description="Set Relax delay (µs).",
    )


class StirapDetuningReport(RecipeReport):
    """Applied readout diagnostics and a sampled f peak on one population basis."""

    f_population_estimate: float | None = Field(
        default=None,
        description="Peak corrected f fraction (unitless and unclipped) on the sampled grid; null when unavailable.",
    )
    pump_detuning_mhz: float | None = Field(
        default=None,
        description="Pump offset from the measured carrier in MHz at the grid peak; null when unavailable.",
    )
    stokes_detuning_mhz: float | None = Field(
        default=None,
        description="Stokes offset from the measured carrier in MHz at the grid peak; null when unavailable.",
    )

    classified_f_peak: StirapDetuningClassifiedPeak | None = Field(
        default=None,
        description="Largest sampled fraction of shots labeled f (0–1) and its coordinates; null if corrected or unavailable.",
    )

    condition_number: float | None = Field(
        default=None,
        description="2-norm condition number of the applied readout matrix; null without correction or if unrepresentable.",
    )
    minimum_singular_value: float | None = Field(
        default=None,
        description="Smallest singular value of the applied readout matrix; null without correction.",
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
    sigma 0.05 µs and length 0.3 µs. Use a waveform library key or an exact Form
    option such as Inline: cosine, then set pump_waveform.length in µs through
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
            classified_f_peak=analysis.result.classified_f_peak,
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
