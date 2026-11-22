"""Readout length sweep scored by the thermal-chain score.

Recipe points sets ADC window › Expts (count) in the GUI.
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
        description="Select the library module in Readout pulse and ADC window; omit to keep the Form source.",
    )
    length_start_us: float | None | Keep = Field(
        KEEP,
        description="Set ADC window › Start (µs), the first setting in the scan.",
    )
    length_stop_us: float | None | Keep = Field(
        KEEP,
        description="Set ADC window › Stop (µs), the last setting in the scan.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Set ADC window › Expts (count); enter at least 2 settings.",
    )
    relax_delay: float | None | Keep = Field(
        KEEP,
        description="Set Thermal wait K (µs), the relaxation wait after each 16-read shot.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), the 16-read shots per setting in each round.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), the number of acquisitions per window.",
    )


class RoOptimizeChainLengthReport(ChainSweepReport):
    """ADC window with the largest combined heuristic score; cancellation returns no report."""

    length_us: float | None = Field(
        default=None,
        description="Selected ADC window (µs); null if analysis gave no result.",
    )


@recipe(
    "ro_optimize_chain_length",
    description="Sweep the ADC window with thermally reset 16-read shots. The readout pulse lasts the window plus Length margin. Score each window and save the sums and analysis.",
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

    Set ADC window › Expts (count) with points. Set Length margin (µs)
    through overrides to change the time added to each ADC window.
    Omission keeps the Form value; null goes to the field owner.
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
