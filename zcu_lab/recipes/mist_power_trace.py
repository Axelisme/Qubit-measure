"""Acquire and save an averaged-IQ MIST probe-gain trace."""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)

from zcu_lab.v2.mist.power_dep.single_trace.gui import MistTraceAnalyzeResult


class MistPowerTraceInputs(RecipeInputs):
    """Inputs for mist power trace; all units and null policies are declared per
    field."""

    probe_ref: str | None = Field(
        None, description="Probe library key; null keeps the GUI selection."
    )
    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the GUI selection. Frequency uses a valid captured library value, then resonator calibration.",
    )

    g0: str | None = Field(
        None,
        description="Ground IQ (empty = first point); null keeps the GUI analysis value. Analysis only.",
    )
    e0: str | None = Field(
        None,
        description="Excited IQ (optional y-limit); null keeps the GUI analysis value. Analysis only.",
    )
    ac_coeff: float | None = Field(
        None,
        description="Photon number / gain squared; null keeps the GUI analysis value. Analysis only.",
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
    Analysis controls: g0, e0, ac_coeff. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    for name, value in (
        ("target", inputs.target),
        ("reuse_tab_id", inputs.reuse_tab_id),
        ("probe_ref", inputs.probe_ref),
        ("readout_ref", inputs.readout_ref),
    ):
        if value is not None and (not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    tab = session.open_recipe_tab(
        "mist/power_dep/single_trace", inputs, target_alias="qubit"
    )
    tab.select(("modules", "probe_pulse"), ref=inputs.probe_ref)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select_sources(inputs)
    edits: list[FieldEdit] = []
    tab.apply_inputs(
        inputs,
        {("sweep", "gain"): None, ("reps",): None, ("rounds",): None},
        edits=edits,
    )
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
                if value is not None
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
