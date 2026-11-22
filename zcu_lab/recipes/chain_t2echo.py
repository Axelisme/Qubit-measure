"""T2 Hahn echo with a K-cycle readout chain at every delay.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from pydantic import Field
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

from zcu_lab.v2.chain._joint import JointResidualSummary
from zcu_lab.v2.chain.t2echo.gui import ChainT2EchoSummary


class ChainT2EchoInputs(RecipeInputs):
    """Inputs for chain T2 echo; units and null policies are per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Set Readout to a library key for the single-tone pulse readout played twice per cycle block.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Set π pulse to the calibrated library pulse.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Set π/2 pulse to the calibrated library pulse.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Set Total echo wait › Stop (µs), the last total echo wait.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Set Total echo wait › Expts (count), the number of sweep points.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), the number of cycles at every sweep point in one round; at least 66.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), the planned number of passes through the sweep.",
    )


class ChainT2EchoReport(RecipeReport, JointResidualSummary):
    """Echo estimates and scalar joint residuals; cancellation returns no report."""

    t2_us: float | None = Field(description="T2 echo (µs); null if unavailable.")
    t2_err_us: float | None = Field(
        description="T2 echo Wald standard error (µs); null if unavailable."
    )
    tf_us: float | None = Field(
        description="Third-state decay time (µs); null if unavailable."
    )
    tf_err_us: float | None = Field(
        description="Third-state decay time Wald standard error (µs); null if unavailable."
    )
    first_read_t2_us: float | None = Field(
        description="First-read T2 echo (µs); null if unavailable."
    )


@recipe(
    "chain_t2echo",
    description="Sweep total echo wait, fit all 65 K-cycle features jointly, and save the sums, decay times and scalar residual diagnostics without writeback.",
)
async def chain_t2echo(
    session: RecipeSession, inputs: ChainT2EchoInputs
) -> ChainT2EchoReport:
    """Sweep the total echo wait with one K-cycle per rep: K, then (pi/2, x/2, pi, x/2, pi/2, read, read) x 8.

    relax_delay is the thermalisation wait K; the Form starts it at 5 T1 and
    overrides can change it. Primary fits the first read with the twotone
    echo decay fit for the starting value, then fits all 65 numbers jointly.
    It reports T2 echo and the third-state decay time with Wald errors, J,
    df, J/df, its chi-square reference limits and the fraction of whitened
    components with |r| > 3. The per-delay projection stays in the figure.
    The diagnostics do not decide acceptance. Save keeps the running sums
    and analysis. No writeback is proposed or requested.

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
                J=None,
                df=None,
                j_per_df=None,
                expected_band_low=None,
                expected_band_high=None,
                tail_fraction=None,
                first_read_t2_us=None,
                analysis_failure=analysis.reason,
            )
        return ChainT2EchoReport(
            t2_us=report_number(analysis.result.t2),
            t2_err_us=report_number(analysis.result.t2_err),
            tf_us=report_number(analysis.result.tf),
            tf_err_us=report_number(analysis.result.tf_err),
            J=analysis.result.J,
            df=analysis.result.df,
            j_per_df=analysis.result.j_per_df,
            expected_band_low=analysis.result.expected_band_low,
            expected_band_high=analysis.result.expected_band_high,
            tail_fraction=analysis.result.tail_fraction,
            first_read_t2_us=report_number(analysis.result.first_read_t2),
        )
    finally:
        tab.save()


DEFINITION = chain_t2echo
