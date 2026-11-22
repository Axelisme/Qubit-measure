"""Acquire and save an averaged-IQ MIST probe-gain trace."""

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
)


class MistPowerTraceInputs(RecipeInputs):
    """Inputs for mist power trace; all units and null policies are declared per
    field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    probe_ref: str | None | Keep = Field(
        KEEP,
        description="Resonator probe library key. Omit to keep; recipe accepts null, but Form rejects it before Run.",
    )
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key. Omit to keep; recipe accepts null, but Form rejects it before Run.",
    )

    g0: str | None | Keep = Field(
        KEEP,
        description="Analyze › Ground IQ reference (a.u., e.g. 1+2j). Omit to keep; null: mean of first 5% (≥1 point).",
    )
    e0: str | None | Keep = Field(
        KEEP,
        description="Analyze › Excited IQ (optional y-limit), e.g. 1+2j. Omit to keep; null restores automatic limits.",
    )
    ac_coeff: float | None | Keep = Field(
        KEEP,
        description="Analyze › Photon number / gain squared (photons/a.u.²). Omit to keep; null uses Probe gain (a.u.).",
    )


class MistPowerTraceReport(RecipeReport):
    """MIST power trace results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default displacement figure completed without a fit."
    )

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_power_trace",
    description="Sweep Resonator probe gain, plot how far mean IQ moves from its reference, and save the mean IQ data and Analyze figure without a fit or writeback.",
)
async def mist_power_trace(
    session: RecipeSession, inputs: MistPowerTraceInputs
) -> MistPowerTraceReport:
    """Probe across gain, save mean complex IQ and plot its displacement.

    No n → expts input: Probe gain sweep supplies the gain grid; use overrides to edit it.
    Omitted Analyze inputs keep their current settings; null selects the scan-start IQ
    reference, automatic y limits, or gain coordinates, respectively.
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
