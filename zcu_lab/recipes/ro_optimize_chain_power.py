"""Readout gain sweep scored by the readout-chain metric.

Recipe points → expts: the gain sample count maps to the GUI Gain Expts field.
"""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2.twotone.ro_optimize.chain_common import ChainAnalysis, SettingFailure


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


class RoOptimizeChainPowerReport(RecipeReport):
    """Gain with the largest readout-chain metric; cancellation returns no report."""

    gain: float | None = Field(
        description="Gain with the largest metric in a.u.; null if analysis gave no result."
    )
    metric: float | None = Field(
        description="Read-drive-read information at that gain; null if unavailable."
    )
    metric_err: float | None = Field(
        description="One-sigma error of that metric; null if unavailable."
    )
    separation: float | None = Field(
        description="d/sigma at that gain; null if unavailable."
    )
    retention: list[float | None] = Field(
        description="λ of its two most populated states; empty if unavailable."
    )
    third_population: float | None = Field(
        description="Third-state population at that gain; null if unavailable."
    )
    failures: list[SettingFailure] = Field(
        description="Gains without a metric, with the reason."
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
    wait. Primary computes the read-drive-read information of each gain from
    the saved sums and reports the gain with the largest value, its error,
    d/sigma, λ and third-state population. No writeback is proposed or
    requested. Save keeps the moment sums and the analysis.

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
            return RoOptimizeChainPowerReport(
                gain=None,
                metric=None,
                metric_err=None,
                separation=None,
                retention=[],
                third_population=None,
                failures=[],
                analysis_failure=analysis.reason,
            )
        best = analysis.result
        return RoOptimizeChainPowerReport(
            gain=report_number(best.best_setting),
            metric=report_number(best.metric),
            metric_err=report_number(best.metric_err),
            separation=report_number(best.separation),
            retention=[report_number(value) for value in best.retention],
            third_population=report_number(best.third_population),
            failures=list(best.failures),
        )
    finally:
        tab.save()


DEFINITION = ro_optimize_chain_power
