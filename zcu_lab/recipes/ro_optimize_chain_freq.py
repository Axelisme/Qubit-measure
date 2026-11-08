"""Readout frequency sweep scored by the readout-chain metric.

Recipe points → expts: the frequency sample count maps to the GUI Readout freq Expts field.
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


class RoOptimizeChainFreqReport(ChainSweepReport):
    """Frequency with the largest readout-chain metric; cancellation returns no report."""

    freq_mhz: float | None = Field(
        default=None,
        description="Frequency with the largest metric in MHz; null if analysis gave no result.",
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
            return RoOptimizeChainFreqReport(analysis_failure=analysis.reason)
        return RoOptimizeChainFreqReport(
            freq_mhz=report_number(analysis.result.best.setting),
            **chain_report_fields(analysis.result),
        )
    finally:
        tab.save()


DEFINITION = ro_optimize_chain_freq
