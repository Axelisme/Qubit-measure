"""T1 with a K-cycle readout chain at every delay.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from pydantic import Field
from zcu_tools.analysis.readout_chain import CycleCheck
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    Overrides,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    ReuseTabId,
    SingleTarget,
    recipe,
    report_number,
)

from zcu_lab.v2.chain.t1.gui import ChainT1Summary


class ChainT1Inputs(RecipeInputs):
    """Inputs for chain T1; units and null policies are per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi-pulse library key.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Delay sweep stop in us.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Sweep sample count.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Cycles per delay in each round, at least 66.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Round count; the live plot updates after each round.",
    )


class ChainT1Report(RecipeReport):
    """Chain T1 fit and its checks; cancellation returns no report."""

    t1_us: float | None = Field(
        description="T1 in us, the larger-amplitude eigen time; null means not estimable."
    )
    t1_err_us: float | None = Field(
        description="T1 uncertainty in us; null means not estimable."
    )
    secondary_time_us: float | None = Field(
        description="The other eigen time in us; null means not estimable."
    )
    secondary_time_err_us: float | None = Field(
        description="Secondary time uncertainty in us; null means not estimable."
    )
    within_point: CycleCheck | None = Field(
        description="Within-point check J, df, p; null without analysis."
    )
    sharing: CycleCheck | None = Field(
        description="Sharing check J, df, p null; null without analysis."
    )
    curve: CycleCheck | None = Field(
        description="Curve check J, df, p; null without analysis."
    )


@recipe(
    "chain_t1",
    description="Run one K-cycle readout-chain T1, fit all 65 numbers, save sums and analysis.",
)
async def chain_t1(session: RecipeSession, inputs: ChainT1Inputs) -> ChainT1Report:
    """Sweep the wait after a pi pulse with one K-cycle per rep: K, then (pi, wait, read, read) x 8.

    relax_delay is the thermalisation wait K; the Form starts it at 5 T1 and
    overrides can change it. Primary fits the first read with a single decay
    for the starting value, then fits all 65 numbers jointly with the
    rate-equation model. It reports T1, the eigen time with the larger
    amplitude, and the other eigen time as the secondary time, with Wald errors
    and three checks. No writeback is proposed or requested. Save keeps the
    running sums and the analysis.

    points → expts sets the sweep sample count.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "chain/t1",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("pi_pulse", inputs.pi_ref)
    tab.set("delay.stop", inputs.max_delay_us)
    tab.set("delay.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ChainT1Summary)
        if analysis.result is None:
            return ChainT1Report(
                t1_us=None,
                t1_err_us=None,
                secondary_time_us=None,
                secondary_time_err_us=None,
                within_point=None,
                sharing=None,
                curve=None,
                analysis_failure=analysis.reason,
            )
        return ChainT1Report(
            t1_us=report_number(analysis.result.t1),
            t1_err_us=report_number(analysis.result.t1_err),
            secondary_time_us=report_number(analysis.result.secondary_time),
            secondary_time_err_us=report_number(analysis.result.secondary_time_err),
            within_point=analysis.result.within_point,
            sharing=analysis.result.sharing,
            curve=analysis.result.curve,
        )
    finally:
        tab.save()


DEFINITION = chain_t1
