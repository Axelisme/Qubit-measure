"""Readout frequency sweep scored by the thermal-chain score.

Recipe points → expts: the frequency sample count maps to the GUI Readout freq Expts field.
"""

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    Overrides,
    RecipeInputs,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2.twotone.ro_optimize._chain_common import ChainAnalysis

from .ro_optimize_chain_helpers import ChainSweepReport, chain_report_fields


class RoOptimizeChainFreqInputs(RecipeInputs):
    """Inputs for the readout-chain frequency sweep; units and null policies are per field."""

    overrides: Overrides
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
    relax_delay: float | None | Keep = Field(
        KEEP, description="K: thermal relaxation between 16-read shots, in us."
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="16-read shots per setting in each round, at least 1.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Round count; the live metric updates between rounds.",
    )


class RoOptimizeChainFreqReport(ChainSweepReport):
    """Frequency with the largest combined heuristic score; cancellation returns no report."""

    freq_mhz: float | None = Field(
        default=None,
        description="Frequency with the largest combined heuristic score in MHz; null if analysis gave no result.",
    )


@recipe(
    "ro_optimize_chain_freq",
    description="Sweep readout frequency with thermally reset 16-read shots, score each frequency by the thermal-chain score, save sums and analysis.",
)
async def ro_optimize_chain_freq(
    session: RecipeSession, inputs: RoOptimizeChainFreqInputs
) -> RoOptimizeChainFreqReport:
    """Sweep frequency with thermal relaxation K and 16 reads per shot, no drive.

    Primary selects the maximum combined score, keeping acquisition order on
    exact ties. It reports score and QND with model-conditional approximate
    error scales, correlation, amplification and M (correlation growth, not
    identifiable physical leakage). Errors do not affect selection.
    Save keeps shot counts, all 107 sums and analysis. No writeback is proposed.

    points → expts sets the frequency sample count. Settings are MHz.
    relax_delay sets K in us. Omission retains the Form value; null goes to its owner.
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
    tab.set("relax_delay", inputs.relax_delay)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
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
