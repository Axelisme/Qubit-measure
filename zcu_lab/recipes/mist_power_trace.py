"""Acquire and save an averaged-IQ MIST probe-gain trace."""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)

from zcu_lab.v2.mist.power_dep.single_trace.gui import MistTraceAnalyzeResult


class MistPowerTraceInputs(RecipeInputs):
    """Inputs for mist power trace; all units and null policies are declared per
    field."""

    probe_ref: str | None | Keep = Field(
        KEEP,
        description="Probe library key; omission retains the Form value, null clears/disables it.",
    )
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value, null clears/disables it.",
    )

    g0: str | None | Keep = Field(
        KEEP,
        description="Ground IQ (empty = first point); omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    e0: str | None | Keep = Field(
        KEEP,
        description="Excited IQ (optional y-limit); omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    ac_coeff: float | None | Keep = Field(
        KEEP,
        description="Photon number / gain squared; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class MistPowerTraceReport(RecipeReport):
    """MIST power trace results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default displacement figure completed without a fit."
    )

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_power_trace",
    description="Acquire MIST mean IQ versus probe gain, save raw data and plot.",
)
async def mist_power_trace(
    session: RecipeSession, inputs: MistPowerTraceInputs
) -> MistPowerTraceReport:
    """Probe across gain, save mean complex IQ and plot its displacement.

    No n → expts input: the captured Form supplies the gain grid.
    Analysis controls: g0, e0, ac_coeff. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "mist/power_dep/single_trace",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("modules.probe_pulse", inputs.probe_ref)
    tab.select("modules.readout", inputs.readout_ref)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=MistTraceAnalyzeResult,
            updates={
                key: value
                for key, value in {
                    "g0": inputs.g0,
                    "e0": inputs.e0,
                    "ac_coeff": inputs.ac_coeff,
                }.items()
                if not isinstance(value, Keep)
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
