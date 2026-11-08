"""Readout length sweep scored by the readout-chain metric.

Recipe points → expts: the window sample count maps to the GUI Readout length Expts field.
"""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2.twotone.ro_optimize._chain_common import ChainAnalysis

from .ro_optimize_chain_helpers import ChainSweepReport, chain_report_fields


class RoOptimizeChainLengthInputs(RecipeInputs):
    """Inputs for the readout-chain length sweep; units and null policies are per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    length_start_us: float | None | Keep = Field(
        KEEP,
        description="Shortest ADC window in us; each pulse is its window plus 0.11 us.",
    )
    length_stop_us: float | None | Keep = Field(
        KEEP,
        description="Longest ADC window in us.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Window sample count, at least 2.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Blocks of 32 reads per window in each round, at least 2.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Rounds per window; the live metric updates between rounds.",
    )


class RoOptimizeChainLengthReport(ChainSweepReport):
    """ADC window with the largest readout-chain metric; cancellation returns no report."""

    length_us: float | None = Field(
        default=None,
        description="ADC window with the largest metric in us; null if analysis gave no result.",
    )


@recipe(
    "ro_optimize_chain_length",
    description="Sweep the readout ADC window, with the pulse 0.11 us longer, in 32-read blocks; score each window by the readout-chain metric, save sums and analysis.",
)
async def ro_optimize_chain_length(
    session: RecipeSession, inputs: RoOptimizeChainLengthInputs
) -> RoOptimizeChainLengthReport:
    """Read 32 times in a row at each readout length, with no qubit pulse.

    Each shot is a block of 32 consecutive reads with no reset, pulse or
    wait; at each ADC window the readout pulse is that window plus 0.11 us,
    so the read duration changes with the window. Primary
    computes the read-drive-read information of each window from the saved
    sums and reports the window with the largest value, its error, d/sigma,
    λ and third-state population. No writeback is proposed or requested.
    Save keeps the moment sums and the analysis.

    points → expts sets the window sample count. Windows are us.
    Omission retains the Form value; null is passed to its owner.
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
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
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
