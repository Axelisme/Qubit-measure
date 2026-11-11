"""Run, analyze and save a common-amplitude STIRAP hardware sweep."""

from pydantic import Field
from zcu_tools.analysis.singleshot import Centroid3Summary
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)


class StirapAmplitudeInputs(RecipeInputs):
    """Measured carriers, signed base gains and a dimensionless scale range.

    n maps to expts. Waveform references select a library key or an inline
    Form option such as Inline: cosine; overrides edit that family's fields.
    Omitted optional fields retain Form values; null is passed to the owner.
    """

    pump_freq_mhz: float = Field(description="Measured Pump 0–3 carrier in MHz.")
    stokes_freq_mhz: float = Field(description="Measured Stokes 3–2 carrier in MHz.")
    pump_gain: float = Field(description="Signed Pump base gain, multiplied by scale.")
    stokes_gain: float = Field(
        description="Signed Stokes base gain, multiplied by scale."
    )
    start: float = Field(description="First common gain multiplier, dimensionless.")
    stop: float = Field(description="Last common gain multiplier, dimensionless.")
    n: int = Field(ge=2, description="Scale samples, mapped to expts.")
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
        description="Fixed Stokes lead/us, positive starts Stokes first; omitted retains, null reaches owner.",
    )
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key with centroid3 calibration; omitted retains, null reaches owner.",
    )
    shots: int | None | Keep = Field(
        KEEP,
        description="Repetitions at each scale in one round; omitted retains, null reaches owner.",
    )
    relax_delay_us: float | None | Keep = Field(
        KEEP,
        description="Relaxation after every shot in us; omitted retains, null reaches owner.",
    )


class StirapAmplitudeReport(RecipeReport):
    """Saved-readout diagnostics, not a transfer fidelity or writeback proposal."""

    condition_number: float | None = Field(
        description="Saved training confusion condition number; null if analysis is unavailable."
    )
    minimum_singular_value: float | None = Field(
        description="Saved training confusion minimum singular value; null if analysis is unavailable."
    )


@recipe(
    "stirap_amplitude",
    description="Sweep both STIRAP gains with one hardware multiplier, analyze g/e/f and save.",
)
async def stirap_amplitude(
    session: RecipeSession, inputs: StirapAmplitudeInputs
) -> StirapAmplitudeReport:
    """Run one hardware gain-scale loop at fixed carriers, lead and envelopes.

    n → expts controls scale points. Pump/Stokes gain signs are retained.
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
    tab.set("pump_gain", inputs.pump_gain)
    tab.set("stokes_gain", inputs.stokes_gain)
    tab.set("stokes_lead", inputs.stokes_lead_us)
    tab.set("scale.start", inputs.start)
    tab.set("scale.stop", inputs.stop)
    tab.set("scale.n", inputs.n)
    tab.set("shots", inputs.shots)
    tab.set("relax_delay", inputs.relax_delay_us)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=Centroid3Summary)
        if analysis.result is None:
            return StirapAmplitudeReport(
                condition_number=None,
                minimum_singular_value=None,
                analysis_failure=analysis.reason,
            )
        return StirapAmplitudeReport(
            condition_number=report_number(analysis.result.condition_number),
            minimum_singular_value=report_number(
                analysis.result.minimum_singular_value
            ),
        )
    finally:
        tab.save()


DEFINITION = stirap_amplitude
