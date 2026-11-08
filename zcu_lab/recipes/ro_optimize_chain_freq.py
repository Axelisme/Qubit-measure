"""Readout frequency sweep scored by the readout-chain metric.

Recipe points → expts: the frequency sample count maps to the GUI Readout freq Expts field.
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


class RoOptimizeChainFreqInputs(RecipeInputs):
    """Inputs for the readout-chain frequency sweep; units and null policies are per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    freq_start_mhz: float | None | Keep = Field(
        KEEP,
        description="First readout frequency in MHz.",
    )
    freq_stop_mhz: float | None | Keep = Field(
        KEEP,
        description="Last readout frequency in MHz.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Frequency sample count, at least 2.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Blocks of 32 reads per frequency in each round, at least 2.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Round count; the live metric updates between rounds.",
    )


class RoOptimizeChainFreqReport(RecipeReport):
    """Frequency with the largest readout-chain metric; cancellation returns no report."""

    freq_mhz: float | None = Field(
        description="Frequency with the largest metric in MHz; null if analysis gave no result."
    )
    metric: float | None = Field(
        description="Read-drive-read information at that frequency; null if unavailable."
    )
    metric_err: float | None = Field(
        description="One-sigma error of that metric; null if unavailable."
    )
    separation: float | None = Field(
        description="d/sigma at that frequency; null if unavailable."
    )
    retention: list[float | None] = Field(
        description="λ of its two most populated states; empty if unavailable."
    )
    third_population: float | None = Field(
        description="Third-state population at that frequency; null if unavailable."
    )
    failures: list[SettingFailure] = Field(
        description="Frequencies (MHz) without a metric, with the reason."
    )


@recipe(
    "ro_optimize_chain_freq",
    description="Sweep readout frequency with 32-read blocks, score each frequency by the readout-chain metric, save sums and analysis.",
)
async def ro_optimize_chain_freq(
    session: RecipeSession, inputs: RoOptimizeChainFreqInputs
) -> RoOptimizeChainFreqReport:
    """Read 32 times in a row at each readout frequency, with no qubit pulse.

    Each shot is a block of 32 consecutive reads with no reset, pulse or
    wait. Primary computes the read-drive-read information of each frequency from
    the saved sums and reports the frequency with the largest value, its error,
    d/sigma, λ and third-state population. No writeback is proposed or
    requested. Save keeps the moment sums and the analysis.

    points → expts sets the frequency sample count. Frequencies are MHz.
    Omission retains the Form value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/ro_optimize/chain_freq",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.set("freq.start", inputs.freq_start_mhz)
    tab.set("freq.stop", inputs.freq_stop_mhz)
    tab.set("freq.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ChainAnalysis)
        if analysis.result is None:
            return RoOptimizeChainFreqReport(
                freq_mhz=None,
                metric=None,
                metric_err=None,
                separation=None,
                retention=[],
                third_population=None,
                failures=[],
                analysis_failure=analysis.reason,
            )
        best = analysis.result
        return RoOptimizeChainFreqReport(
            freq_mhz=report_number(best.best_setting),
            metric=report_number(best.metric),
            metric_err=report_number(best.metric_err),
            separation=report_number(best.separation),
            retention=[report_number(value) for value in best.retention],
            third_population=report_number(best.third_population),
            failures=list(best.failures),
        )
    finally:
        tab.save()


DEFINITION = ro_optimize_chain_freq
