"""T2 Hahn echo with a K-cycle readout chain at every delay.

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
    recipe,
    report_number,
)

from zcu_lab.v2.chain.t2echo.gui import ChainT2EchoSummary


class ChainT2EchoInputs(RecipeInputs):
    """Inputs for chain T2 echo; units and null policies are per field."""

    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi-pulse library key.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi/2-pulse library key.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Total echo-delay stop in us, not half-delay.",
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


class ChainT2EchoReport(RecipeReport):
    """Chain echo fit and its checks; cancellation returns no report."""

    t2_us: float | None = Field(
        description="T2 echo from the chain fit in us; null means not estimable."
    )
    t2_err_us: float | None = Field(
        description="T2 echo uncertainty in us; null means not estimable."
    )
    tf_us: float | None = Field(
        description="Third-state decay time in us; null means not estimable."
    )
    tf_err_us: float | None = Field(
        description="Third-state decay uncertainty in us; null means not estimable."
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
    first_read_t2_us: float | None = Field(
        description="T2 echo in us of the first-read twotone fit; null means not estimable."
    )


@recipe(
    "chain_t2echo",
    description="Run one K-cycle readout-chain T2 echo, fit all 65 numbers, save sums and analysis.",
)
async def chain_t2echo(
    session: RecipeSession, inputs: ChainT2EchoInputs
) -> ChainT2EchoReport:
    """Sweep the total echo wait with one K-cycle per rep: K, then (pi/2, x/2, pi, x/2, pi/2, read, read) x 8.

    relax_delay is the thermalisation wait K; the Form starts it at 5 T1 and
    overrides can change it. Primary fits the first read with the twotone
    echo decay fit for the starting value, then fits all 65 numbers jointly.
    It reports T2 echo and the third-state decay time with Wald errors and
    three checks. No writeback is proposed or requested. Save keeps the
    running sums and the analysis.

    points → expts sets the sweep sample count.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "chain/t2echo",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("pi_pulse", inputs.pi_ref)
    tab.select("pi2_pulse", inputs.pi2_ref)
    tab.set("delay.stop", inputs.max_delay_us)
    tab.set("delay.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ChainT2EchoSummary)
        if analysis.result is None:
            return ChainT2EchoReport(
                t2_us=None,
                t2_err_us=None,
                tf_us=None,
                tf_err_us=None,
                within_point=None,
                sharing=None,
                curve=None,
                first_read_t2_us=None,
                analysis_failure=analysis.reason,
            )
        return ChainT2EchoReport(
            t2_us=report_number(analysis.result.t2),
            t2_err_us=report_number(analysis.result.t2_err),
            tf_us=report_number(analysis.result.tf),
            tf_err_us=report_number(analysis.result.tf_err),
            within_point=analysis.result.within_point,
            sharing=analysis.result.sharing,
            curve=analysis.result.curve,
            first_read_t2_us=report_number(analysis.result.first_read_t2),
        )
    finally:
        tab.save()


DEFINITION = chain_t2echo
