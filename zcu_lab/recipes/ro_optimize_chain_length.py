"""Readout length sweep scored by the thermal-chain score.

Recipe points → expts: the window sample count maps to the GUI Readout length Expts field.
"""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    Overrides,
    RecipeInputs,
    RecipeSession,
    ReuseTabId,
    SingleTarget,
    recipe,
    report_number,
)

from zcu_lab.v2.twotone.ro_optimize._chain_common import ChainAnalysis

from .ro_optimize_chain_helpers import ChainSweepReport, chain_report_fields


class RoOptimizeChainLengthInputs(RecipeInputs):
    """Inputs for the readout-chain length sweep; units and null policies are per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    length_start_us: float | None | Keep = Field(
        KEEP,
        description="Shortest ADC window in us.",
    )
    length_stop_us: float | None | Keep = Field(
        KEEP,
        description="Longest ADC window in us.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Window sample count, at least 2.",
    )
    relax_delay: float | None | Keep = Field(
        KEEP, description="K: thermal relaxation between 16-read shots, in us."
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="16-read shots per setting in each round, at least 1.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Rounds per window; the live metric updates between rounds.",
    )


class RoOptimizeChainLengthReport(ChainSweepReport):
    """ADC window with the largest combined heuristic score; cancellation returns no report."""

    length_us: float | None = Field(
        default=None,
        description="ADC window with the largest combined heuristic score in us; null if analysis gave no result.",
    )


@recipe(
    "ro_optimize_chain_length",
    description="Sweep the ADC window in thermally reset 16-read shots. Set the pulse length for each window using trigger offset and flight time, then score each window and save the sums and analysis.",
)
async def ro_optimize_chain_length(
    session: RecipeSession, inputs: RoOptimizeChainLengthInputs
) -> RoOptimizeChainLengthReport:
    """Sweep ADC window with thermal relaxation K and 16 reads per shot, no drive.

    Primary selects the maximum combined score, keeping acquisition order on
    exact ties. It reports score and QND with model-conditional approximate
    error scales, correlation, amplification and M (correlation growth, not
    identifiable physical leakage). Errors do not affect selection.
    Save keeps shot counts, all 107 sums and analysis. No writeback is proposed.

    points → expts sets the number of ADC windows. In us, each pulse lasts
    trig_offset minus time_of_fly plus the window and 0.11 us.
    relax_delay sets K in us. Omission retains the Form value; null goes to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/ro_optimize/chain_length",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.set("length.start", inputs.length_start_us)
    tab.set("length.stop", inputs.length_stop_us)
    tab.set("length.n", inputs.points)
    tab.set("relax_delay", inputs.relax_delay)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ChainAnalysis)
        if analysis.result is None:
            return RoOptimizeChainLengthReport(analysis_failure=analysis.reason)
        return RoOptimizeChainLengthReport(
            length_us=report_number(analysis.result.best.setting),
            **chain_report_fields(analysis.result),
        )
    finally:
        tab.save()


DEFINITION = ro_optimize_chain_length
