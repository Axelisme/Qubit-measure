"""Readout gain sweep scored by the readout-chain metric.

Recipe points → expts: the gain sample count maps to the GUI Gain Expts field.
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


class RoOptimizeChainPowerInputs(RecipeInputs):
    """Inputs for the readout-chain gain sweep; units and null policies are per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    gain_start: float | None | Keep = Field(
        KEEP,
        description="First readout gain in a.u.",
    )
    gain_stop: float | None | Keep = Field(
        KEEP,
        description="Last readout gain in a.u.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Gain sample count, at least 2.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Blocks of 32 reads per gain in each round, at least 2.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Round count; the live metric updates between rounds.",
    )


class RoOptimizeChainPowerReport(ChainSweepReport):
    """Gain with the largest information minus one sigma; cancellation returns no report."""

    gain: float | None = Field(
        default=None,
        description="Gain with the largest information minus one sigma in a.u.; null if analysis gave no result.",
    )


@recipe(
    "ro_optimize_chain_power",
    description="Sweep readout gain with 32-read blocks, score each gain by the readout-chain metric, save sums and analysis.",
)
async def ro_optimize_chain_power(
    session: RecipeSession, inputs: RoOptimizeChainPowerInputs
) -> RoOptimizeChainPowerReport:
    """Read 32 times in a row at each readout gain, with no qubit pulse.

    Each shot is a block of 32 consecutive reads with no reset, pulse or
    wait. Primary computes the all-lag information per read of each gain from
    the saved sums and reports the gain with the largest information minus its
    one-sigma error, with its information, QND, their errors, d/sigma, λ and
    third-state population, plus the gains with no signal and the failed fits.
    No writeback is proposed or requested. Save keeps the moment sums and the
    analysis.

    points → expts sets the gain sample count. Gains are a.u.
    Omission retains the Form value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/ro_optimize/chain_power",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.set("gain.start", inputs.gain_start)
    tab.set("gain.stop", inputs.gain_stop)
    tab.set("gain.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ChainAnalysis)
        if analysis.result is None:
            return RoOptimizeChainPowerReport(analysis_failure=analysis.reason)
        return RoOptimizeChainPowerReport(
            gain=report_number(analysis.result.best.setting),
            **chain_report_fields(analysis.result),
        )
    finally:
        tab.save()


DEFINITION = ro_optimize_chain_power
