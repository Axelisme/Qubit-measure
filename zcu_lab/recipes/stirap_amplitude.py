"""Run, analyze and save an independent Pump-by-Stokes gain hardware sweep."""

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

from zcu_lab.v2.twotone.stirap.amplitude.core import StirapAmplitudeSummary


class StirapAmplitudeInputs(RecipeInputs):
    """Measured carriers and two independent signed gain ranges in a.u.

    pump_n and stokes_n map to their respective sweep expts.
    Waveform references select a library key or an inline
    Form option such as Inline: cosine; overrides edit that family's fields.
    Omitted optional fields retain Form values; null is passed to the owner.
    """

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    pump_freq_mhz: float = Field(description="Measured Pump 0–3 carrier in MHz.")
    stokes_freq_mhz: float = Field(description="Measured Stokes 3–2 carrier in MHz.")
    pump_start: float = Field(
        description="First signed Pump gain in a.u., within [-1, 1]."
    )
    pump_stop: float = Field(
        description="Last signed Pump gain in a.u., within [-1, 1]."
    )
    pump_n: int = Field(
        ge=2, description="Pump gain samples, mapped to pump_gain.expts."
    )
    stokes_start: float = Field(
        description="First signed Stokes gain in a.u., within [-1, 1]."
    )
    stokes_stop: float = Field(
        description="Last signed Stokes gain in a.u., within [-1, 1]."
    )
    stokes_n: int = Field(
        ge=2, description="Stokes gain samples, mapped to stokes_gain.expts."
    )
    pump_ch: int | None | Keep = Field(
        KEEP, description="Pump charge generator; omitted retains, null reaches owner."
    )
    stokes_ch: int | None | Keep = Field(
        KEEP,
        description="Stokes charge generator, distinct from Pump; omitted retains, null reaches owner.",
    )
    pump_waveform_ref: str | None | Keep = Field(
        KEEP,
        description="Pump envelope library key or inline option; omitted retains, null reaches owner.",
    )
    stokes_waveform_ref: str | None | Keep = Field(
        KEEP,
        description="Stokes envelope library key or inline option; omitted retains, null reaches owner.",
    )
    stokes_lead_us: float | None | Keep = Field(
        KEEP,
        description="Fixed Stokes lead/µs, positive starts Stokes first; omitted retains, null reaches owner.",
    )
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key with centroid3 calibration; omitted retains, null reaches owner.",
    )
    shots: int | None | Keep = Field(
        KEEP,
        description="Repetitions at each gain pair in one round; omitted retains, null reaches owner.",
    )
    relax_delay_us: float | None | Keep = Field(
        KEEP,
        description="Relaxation after every shot in µs; omitted retains, null reaches owner.",
    )


class StirapAmplitudeReport(RecipeReport):
    """Saved readout diagnostics and the peak within the measured scan grid."""

    f_population_estimate: float | None = Field(
        default=None,
        description="Largest finite corrected f population (unitless, unclipped) on this grid, not fidelity or a global optimum. Null when unavailable.",
    )
    pump_gain: float | None = Field(
        default=None,
        description="Pump gain in a.u. at the grid peak; null when unavailable.",
    )
    stokes_gain: float | None = Field(
        default=None,
        description="Stokes gain in a.u. at the grid peak; null when unavailable.",
    )

    condition_number: float | None = Field(
        description="Saved training confusion condition number; null if analysis is unavailable."
    )
    minimum_singular_value: float | None = Field(
        description="Saved training confusion minimum singular value; null if analysis is unavailable."
    )


@recipe(
    "stirap_amplitude",
    description="Sweep independent STIRAP Pump and Stokes gains in hardware, analyze g/e/f and save.",
)
async def stirap_amplitude(
    session: RecipeSession, inputs: StirapAmplitudeInputs
) -> StirapAmplitudeReport:
    """Run two independent hardware gain loops at fixed carriers, lead and envelopes.

    pump_n/stokes_n → expts controls the two gain axes. Signed and zero gains play.
    Run compilation rejects pulse gains outside [-1, 1]; Form edits do not.
    Choose gauss, cosine, drag or arb independently; overrides set waveform
    parameters. Analyze only the saved centroid3 readout. Save after successful
    Run even when Analyze fails; failed Run does not Save. No writeback occurs.
    """
    tab = session.open_tab(
        "twotone/stirap/amplitude",
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
    tab.set("pump_gain.start", inputs.pump_start)
    tab.set("pump_gain.stop", inputs.pump_stop)
    tab.set("pump_gain.n", inputs.pump_n)
    tab.set("stokes_gain.start", inputs.stokes_start)
    tab.set("stokes_gain.stop", inputs.stokes_stop)
    tab.set("stokes_gain.n", inputs.stokes_n)
    tab.set("stokes_lead", inputs.stokes_lead_us)
    tab.set("shots", inputs.shots)
    tab.set("relax_delay", inputs.relax_delay_us)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=StirapAmplitudeSummary)
        if analysis.result is None:
            return StirapAmplitudeReport(
                condition_number=None,
                minimum_singular_value=None,
                analysis_failure=analysis.reason,
            )
        return StirapAmplitudeReport(
            f_population_estimate=analysis.result.f_population_estimate,
            pump_gain=analysis.result.pump_gain,
            stokes_gain=analysis.result.stokes_gain,
            condition_number=report_number(analysis.result.condition_number),
            minimum_singular_value=report_number(
                analysis.result.minimum_singular_value
            ),
        )
    finally:
        tab.save()


DEFINITION = stirap_amplitude
