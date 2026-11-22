"""Acquire and save an averaged-IQ MIST probe-gain trace."""

from pydantic import ConfigDict, Field
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


class MistPowerTraceInputs(RecipeInputs):
    """Choose the MIST gain trace target, pulse modules and Analyze display."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": "Omit Form inputs to keep their values. Null works only as described for each input."
        }
    )

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    probe_ref: str | None | Keep = Field(
        KEEP,
        description="Resonator probe library key. Omit to keep the selection. Recipe parses null; Form rejects it.",
    )
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key. Omit to keep the selection. Recipe parses null; Form rejects it.",
    )

    g0: str | None | Keep = Field(
        KEEP,
        description="Set Analyze › Reference IQ (a.u.) as complex text, e.g. 1+2j. Null uses the first gain point.",
    )
    e0: str | None | Keep = Field(
        KEEP,
        description="Set Analyze › Comparison IQ (a.u.) as complex text. Null uses automatic y-limits.",
    )
    ac_coeff: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Photon number / gain squared. Omit to retain it; null keeps the gain axis in a.u.",
    )


class MistPowerTraceReport(RecipeReport):
    """MIST power trace results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default displacement figure completed without a fit."
    )

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_power_trace",
    description="Measure the mean IQ shift versus resonator probe gain. Analyze the trace and save gain, complex IQ and available shot moments without writeback.",
)
async def mist_power_trace(
    session: RecipeSession, inputs: MistPowerTraceInputs
) -> MistPowerTraceReport:
    """Probe across gain, save mean complex IQ and plot its displacement.

    Omit probe_ref or readout_ref to keep the corresponding Form selection.
    Recipe input parsing accepts explicit null and passes it unchanged to the Form.
    The Form requires both modules and rejects null before Run.

    Probe gain sweep supplies Start, Stop and Expts. Use overrides for a different grid or averaging count.
    The Analyze inputs change the displayed reference or axes, not the acquisition.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "mist/power_dep/single_trace",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("probe_pulse", inputs.probe_ref)
    tab.select("readout", inputs.readout_ref)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=dict,
            updates={
                "g0": inputs.g0,
                "e0": inputs.e0,
                "ac_coeff": inputs.ac_coeff,
            },
        )
        if analysis.result is None:
            return MistPowerTraceReport(
                analysis_completed=False, analysis_failure=analysis.reason
            )
        return MistPowerTraceReport(analysis_completed=True)
    finally:
        tab.save()


DEFINITION = mist_power_trace
