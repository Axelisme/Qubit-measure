"""Acquire and save an averaged-IQ MIST probe-gain trace."""

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe


class MistPowerTraceInputs(FormModel):
    """Inputs for mist power trace; all units and null policies are declared per
    field."""

    target: str | None = Field(
        None,
        description="Qubit alias for this Reset; null uses the GUI aliases without changing focus.",
    )
    reuse_tab_id: str | None = Field(
        None,
        description="Idle matching tab to reset; null opens a new tab. May be combined with target.",
    )
    probe_ref: str | None = Field(
        None, description="Probe library key; null keeps the GUI selection."
    )
    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the GUI selection. Frequency uses a valid captured library value, then resonator calibration.",
    )


class MistPowerTraceReport(FormModel):
    """MIST power trace results and failure reason; cancellation returns no report."""

    analysis_completed: bool = Field(
        description="Whether the default displacement figure completed without a fit."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class MistPowerTraceSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    # This figure-only analysis exposes no scalar fields to the report.


@recipe(
    "mist_power_trace",
    description="Acquire MIST mean IQ versus probe gain, save raw data and plot.",
)
async def mist_power_trace(
    session: RecipeSession, inputs: MistPowerTraceInputs
) -> MistPowerTraceReport:
    """Probe across gain, save mean complex IQ and plot its displacement.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Expected analysis failures return a reason in
    the report; other errors propagate. Save raw data and completed artifacts once
    in the Run finalizer.
    """
    for name, value in (
        ("target", inputs.target),
        ("reuse_tab_id", inputs.reuse_tab_id),
        ("probe_ref", inputs.probe_ref),
        ("readout_ref", inputs.readout_ref),
    ):
        if value is not None and (not value.strip()):
            raise ValueError(f"{name} must be a non-empty string or null")
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "mist/power_dep/single_trace",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select(("modules", "probe_pulse"), ref=inputs.probe_ref)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.set(("sweep", "gain"), None)
    tab.set(("reps",), None)
    tab.set(("rounds",), None)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Analyze the acquisition with the GUI parameters.
        analysis = await tab.analyze("primary", result=MistPowerTraceSummary)
        if analysis.result is None:
            return MistPowerTraceReport(
                analysis_completed=False,
                analysis_failure=analysis.reason,
            )
        return MistPowerTraceReport(analysis_completed=True)
    finally:
        tab.save()


DEFINITION = mist_power_trace
