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

from zcu_lab.v2.twotone.stirap.delay.core import StirapDelaySummary


class StirapDelayInputs(RecipeInputs):
    """Measured carriers in MHz, signed gains and software lead bounds in us.

    Positive lead starts Stokes first. n maps to Form stokes_lead.n and cfg
    stokes_lead.expts. Omitted optional fields keep Form values; null passes to
    their owner unchanged. Waveform refs select library keys or exact Inline
    options; overrides supply the selected waveform's parameters.
    """

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    start_us: float = Field(
        description="First Stokes lead in us; negative starts Pump first."
    )
    stop_us: float = Field(
        description="Last Stokes lead in us; positive starts Stokes first."
    )
    n: int = Field(ge=1, description="Lead point count, mapped to cfg expts.")
    pump_freq_mhz: float = Field(gt=0, description="Measured 0–3 Pump carrier in MHz.")
    stokes_freq_mhz: float = Field(
        gt=0, description="Measured 3–2 Stokes carrier in MHz."
    )
    pump_gain: float = Field(ge=-1, le=1, description="Signed Pump DAC gain.")
    stokes_gain: float = Field(ge=-1, le=1, description="Signed Stokes DAC gain.")
    pump_ch: int | None | Keep = Field(
        default=KEEP,
        description="Pump charge generator; omission keeps Form, null goes to owner.",
    )
    stokes_ch: int | None | Keep = Field(
        default=KEEP,
        description="Different Stokes charge generator; omission keeps Form, null goes to owner.",
    )
    pump_waveform_ref: str | None | Keep = Field(
        default=KEEP,
        description="Pump library waveform key or exact Inline option; omission keeps Form, null goes to owner.",
    )
    stokes_waveform_ref: str | None | Keep = Field(
        default=KEEP,
        description="Stokes library waveform key or exact Inline option; omission keeps Form, null goes to owner.",
    )
    readout_ref: str | None | Keep = Field(
        default=KEEP,
        description="Centroid3 readout library key; omission keeps Form, null goes to owner.",
    )
    shots: int | None | Keep = Field(
        default=KEEP,
        description="Shots per lead; omission keeps Form, null goes to owner.",
    )
    relax_delay_us: float | None | Keep = Field(
        default=KEEP,
        description="Inter-shot relaxation in us; omission keeps Form, null goes to owner.",
    )


class StirapDelayReport(RecipeReport):
    """Saved readout diagnostics and the peak within the measured scan grid."""

    f_population_estimate: float | None = Field(
        default=None,
        description="Largest finite corrected f population (unitless, unclipped) on this grid, not fidelity or a global optimum. Null when unavailable.",
    )
    stokes_lead_us: float | None = Field(
        default=None,
        description="Stokes lead in us at the grid peak; null when unavailable.",
    )

    condition_number: float | None = Field(
        default=None,
        description="Saved confusion 2-norm condition number; null if analysis was cancelled.",
    )
    minimum_singular_value: float | None = Field(
        default=None,
        description="Saved confusion minimum singular value; null if analysis was cancelled.",
    )


@recipe(
    "stirap_delay",
    description="Sweep Stokes lead, analyze saved centroid3 counts and save.",
)
async def stirap_delay(
    session: RecipeSession, inputs: StirapDelayInputs
) -> StirapDelayReport:
    """Play Pump/Stokes at each lead, join, read out, analyze and Save.

    n sets stokes_lead.n, lowered to cfg stokes_lead.expts. Bounds use us,
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
