"""T2 Ramsey with a K-cycle readout chain at every delay.

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

from zcu_lab.v2.chain.t2ramsey.gui import ChainT2RamseySummary


class ChainT2RamseyInputs(RecipeInputs):
    """Inputs for chain T2 Ramsey; units and null policies are per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi/2-pulse library key.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Delay sweep stop in us.",
    )
    detune_mhz: float | None | Keep = Field(
        KEEP,
        description="Second pi/2 phase ramp in MHz (fringes per us of delay).",
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


class ChainT2RamseyReport(RecipeReport):
    """Chain Ramsey fit and its checks; cancellation returns no report."""

    t2_us: float | None = Field(
        description="T2 Ramsey from the chain fit in us; null means not estimable."
    )
    t2_err_us: float | None = Field(
        description="T2 Ramsey uncertainty in us; null means not estimable."
    )
    detuning_mhz: float | None = Field(
        description="Fringe frequency in MHz, ramp included; null means not estimable."
    )
    detuning_err_mhz: float | None = Field(
        description="Fringe frequency uncertainty in MHz; null means not estimable."
    )
    tf_us: float | None = Field(
        description="Third-state decay time in us; null means not estimable."
    )
    tf_err_us: float | None = Field(
        description="Third-state decay uncertainty in us; null means not estimable."
    )
    reliability: float | None = Field(
        description="Diagnostic R of the detuning error, no threshold."
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
        description="T2 in us of the first-read twotone fit; null means not estimable."
    )


@recipe(
    "chain_t2ramsey",
    description="Run one K-cycle readout-chain T2 Ramsey, fit all 65 numbers, save sums and analysis.",
)
async def chain_t2ramsey(
    session: RecipeSession, inputs: ChainT2RamseyInputs
) -> ChainT2RamseyReport:
    """Sweep the Ramsey wait with one K-cycle per rep: K, then (pi/2, wait, pi/2, read, read) x 8.

    The second pi/2 has opposite phase plus the detune ramp. relax_delay is
    the thermalisation wait K; the Form starts it at 5 T1 and overrides can
    change it. Primary fits the first read with the twotone Ramsey fit for
    the starting values, then fits all 65 numbers jointly. It reports T2,
    the fringe frequency and the third-state decay time with Wald errors, R
    and three checks. No writeback is proposed or requested. Save keeps the
    running sums and the analysis.

    points → expts sets the sweep sample count.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "chain/t2ramsey",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("pi2_pulse", inputs.pi2_ref)
    tab.set("delay.stop", inputs.max_delay_us)
    tab.set("delay.n", inputs.points)
    tab.set("detune", inputs.detune_mhz)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ChainT2RamseySummary)
        if analysis.result is None:
            return ChainT2RamseyReport(
                t2_us=None,
                t2_err_us=None,
                detuning_mhz=None,
                detuning_err_mhz=None,
                tf_us=None,
                tf_err_us=None,
                reliability=None,
                within_point=None,
                sharing=None,
                curve=None,
                first_read_t2_us=None,
                analysis_failure=analysis.reason,
            )
        return ChainT2RamseyReport(
            t2_us=report_number(analysis.result.t2),
            t2_err_us=report_number(analysis.result.t2_err),
            detuning_mhz=report_number(analysis.result.detuning),
            detuning_err_mhz=report_number(analysis.result.detuning_err),
            tf_us=report_number(analysis.result.tf),
            tf_err_us=report_number(analysis.result.tf_err),
            reliability=report_number(analysis.result.reliability),
            within_point=analysis.result.within_point,
            sharing=analysis.result.sharing,
            curve=analysis.result.curve,
            first_read_t2_us=report_number(analysis.result.first_read_t2),
        )
    finally:
        tab.save()


DEFINITION = chain_t2ramsey
