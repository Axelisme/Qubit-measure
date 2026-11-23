"""Run, analyze and save an independent STIRAP Stokes-lead scan."""

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

from zcu_lab.v2.twotone.stirap.delay.core import (
    StirapDelayClassifiedPeak,
    StirapDelaySummary,
)


class StirapDelayInputs(RecipeInputs):
    """Measured carriers in MHz, signed gains and software lead bounds in µs.

    Positive lead starts Stokes first. n maps to Form stokes_lead.n and cfg
    stokes_lead.expts. Omitted optional fields keep Form values; null passes to
    their owner unchanged. Waveform refs select library keys or exact Inline
    options; overrides supply the selected waveform's parameters.
    """

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    start_us: float = Field(
        description="Set Stokes lead › Start (µs); negative starts Pump first."
    )
    stop_us: float = Field(
        description="Set Stokes lead › Stop (µs); positive starts Stokes first."
    )
    n: int = Field(ge=1, description="Set Stokes lead › Expts (count).")
    pump_freq_mhz: float = Field(
        gt=0,
        description="Set Pump carrier 0–3 (MHz) to the measured transition frequency.",
    )
    stokes_freq_mhz: float = Field(
        gt=0,
        description="Set Stokes carrier 3–2 (MHz) to the measured transition frequency.",
    )
    pump_gain: float = Field(ge=-1, le=1, description="Set Pump gain (a.u.).")
    stokes_gain: float = Field(ge=-1, le=1, description="Set Stokes gain (a.u.).")
    pump_ch: int | None | Keep = Field(
        default=KEEP,
        description="Set Pump generator (channel).",
    )
    stokes_ch: int | None | Keep = Field(
        default=KEEP,
        description="Set Stokes generator (channel).",
    )
    pump_waveform_ref: str | None | Keep = Field(
        default=KEEP,
        description="Set Pump waveform to a library key or exact inline option.",
    )
    stokes_waveform_ref: str | None | Keep = Field(
        default=KEEP,
        description="Set Stokes waveform to a library key or exact inline option.",
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


class StirapDelayReport(RecipeReport):
    """Applied readout diagnostics and a sampled f peak on one population basis."""

    f_population_estimate: float | None = Field(
        default=None,
        description="Peak corrected f fraction (unitless and unclipped) on the sampled grid; null when unavailable.",
    )
    stokes_lead_us: float | None = Field(
        default=None,
        description="Stokes lead in µs at the grid peak; null when unavailable.",
    )

    classified_f_peak: StirapDelayClassifiedPeak | None = Field(
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
    "stirap_delay",
    description="Sweep Stokes lead, analyze saved centroid3 counts and save.",
)
async def stirap_delay(
    session: RecipeSession, inputs: StirapDelayInputs
) -> StirapDelayReport:
    """Play Pump/Stokes at each lead, join, read out, analyze and Save.

    n sets stokes_lead.n, lowered to cfg stokes_lead.expts. Bounds use µs,
    carriers use MHz, gains retain their sign. Pump/Stokes each select gauss,
    cosine, drag or arb. Optional omission retains Form values; explicit null
    goes unchanged to its owner. Overrides are applied last.
    Failed Run does not Save. Failed/cancelled analysis after Run still Saves
    counts and completed artifacts. No writeback is proposed or requested.
    """
    tab = session.open_tab(
        "twotone/stirap/delay",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("pump_waveform", inputs.pump_waveform_ref)
    tab.select("stokes_waveform", inputs.stokes_waveform_ref)
    tab.set("pump_ch", inputs.pump_ch)
    tab.set("stokes_ch", inputs.stokes_ch)
    tab.set("pump_freq", inputs.pump_freq_mhz)
    tab.set("stokes_freq", inputs.stokes_freq_mhz)
    tab.set("pump_gain", inputs.pump_gain)
    tab.set("stokes_gain", inputs.stokes_gain)
    tab.set("stokes_lead.start", inputs.start_us)
    tab.set("stokes_lead.stop", inputs.stop_us)
    tab.set("stokes_lead.n", inputs.n)
    tab.set("shots", inputs.shots)
    tab.set("relax_delay", inputs.relax_delay_us)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=StirapDelaySummary)
        if analysis.result is None:
            return StirapDelayReport(analysis_failure=analysis.reason)
        return StirapDelayReport(
            f_population_estimate=analysis.result.f_population_estimate,
            classified_f_peak=analysis.result.classified_f_peak,
            stokes_lead_us=analysis.result.stokes_lead_us,
            condition_number=report_number(analysis.result.condition_number),
            minimum_singular_value=report_number(
                analysis.result.minimum_singular_value
            ),
            writeback_applied=False,
        )
    finally:
        tab.save()


DEFINITION = stirap_delay
