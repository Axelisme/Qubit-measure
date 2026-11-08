"""Readout ADC-window sweep scored by the readout-chain metric.

Recipe points → expts: the window sample count maps to the GUI Readout length Expts field.
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


class RoOptimizeChainLengthInputs(RecipeInputs):
    """Inputs for the readout-chain ADC-window sweep; units and null policies are per field."""

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
        description="Longest ADC window in us; the pulse is this plus 0.11 us.",
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


class RoOptimizeChainLengthReport(RecipeReport):
    """ADC window with the largest readout-chain metric; cancellation returns no report."""

    length_us: float | None = Field(
        description="ADC window with the largest metric in us; null if analysis gave no result."
    )
    metric: float | None = Field(
        description="Read-drive-read information at that window; null if unavailable."
    )
    metric_err: float | None = Field(
        description="One-sigma error of that metric; null if unavailable."
    )
    separation: float | None = Field(
        description="d/sigma at that window; null if unavailable."
    )
    retention: list[float | None] = Field(
        description="λ of its two most populated states; empty if unavailable."
    )
    third_population: float | None = Field(
        description="Third-state population at that window; null if unavailable."
    )
    failures: list[SettingFailure] = Field(
        description="Windows (us) without a metric, with the reason."
    )


@recipe(
    "ro_optimize_chain_length",
    description="Sweep the readout ADC window with 32-read blocks, score each window by the readout-chain metric, save sums and analysis.",
)
async def ro_optimize_chain_length(
    session: RecipeSession, inputs: RoOptimizeChainLengthInputs
) -> RoOptimizeChainLengthReport:
    """Read 32 times in a row at each ADC window, with no qubit pulse.

    Each shot is a block of 32 consecutive reads with no reset, pulse or
    wait; the pulse stays at the longest window plus 0.11 us. Primary
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
            return RoOptimizeChainLengthReport(
                length_us=None,
                metric=None,
                metric_err=None,
                separation=None,
                retention=[],
                third_population=None,
                failures=[],
                analysis_failure=analysis.reason,
            )
        best = analysis.result
        return RoOptimizeChainLengthReport(
            length_us=report_number(best.best_setting),
            metric=report_number(best.metric),
            metric_err=report_number(best.metric_err),
            separation=report_number(best.separation),
            retention=[report_number(value) for value in best.retention],
            third_population=report_number(best.third_population),
            failures=list(best.failures),
        )
    finally:
        tab.save()


DEFINITION = ro_optimize_chain_length
