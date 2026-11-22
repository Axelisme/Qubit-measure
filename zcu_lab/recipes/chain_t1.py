"""T1 with a K-cycle readout chain at every delay.

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
from zcu_lab.v2.chain.t1.gui import ChainT1Summary


class ChainT1Inputs(RecipeInputs):
    """Inputs for chain T1; units and null policies are per field."""

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
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Set Wait after π pulse › Stop (µs), the last wait.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Set Wait after π pulse › Expts (count), the number of sweep points.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), the number of cycles at every sweep point in one round; at least 66.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), the planned number of passes through the sweep.",
    )


class ChainT1Report(RecipeReport, JointResidualSummary):
    """Chain T1 estimates and joint residuals; cancellation returns no report."""

    t1_us: float | None = Field(
        description="T1 (µs), the larger-amplitude eigen time; null if unavailable."
    )
    t1_err_us: float | None = Field(
        description="T1 Wald standard error (µs); null if unavailable."
    )
    secondary_time_us: float | None = Field(
        description="Secondary time (µs), the other eigen time; null if unavailable."
    )
    secondary_time_err_us: float | None = Field(
        description="Secondary time Wald standard error (µs); null if unavailable."
    )


@recipe(
    "chain_t1",
    description="Sweep wait after π in K-cycle T1, fit all 65 features jointly, and save the sums, decay times and residual diagnostics without writeback.",
)
async def chain_t1(session: RecipeSession, inputs: ChainT1Inputs) -> ChainT1Report:
    """Sweep the wait after a pi pulse with one K-cycle per rep: K, then (pi, wait, read, read) x 8.

    relax_delay is the thermalisation wait K; the Form starts it at 5 T1 and
    overrides can change it. Primary fits the first read with a single decay
    for the starting value, then fits all 65 numbers jointly with the
    rate-equation model. It reports T1, the eigen time with the larger
    amplitude, and the other eigen time as the secondary time, both in µs
    with Wald standard errors. It reports J, df, J/df, the numerical χ²
    reference band and the fraction of whitened residuals with |r| > 3.
    These numbers do not determine acceptance. The live-direction residual
    series stays in the figure. Save keeps the sums and analysis without writeback.

    points → expts sets the sweep sample count. Omission keeps the Form value;
    explicit null is passed to the Form. Missing Form values stop before
    acquisition. Save keeps completed Run data even when Analyze fails.
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
                J=None,
                df=None,
                j_per_df=None,
                expected_band_low=None,
                expected_band_high=None,
                tail_fraction=None,
                analysis_failure=analysis.reason,
            )
        return ChainT1Report(
            t1_us=report_number(analysis.result.t1),
            t1_err_us=report_number(analysis.result.t1_err),
            secondary_time_us=report_number(analysis.result.secondary_time),
            secondary_time_err_us=report_number(analysis.result.secondary_time_err),
            J=analysis.result.J,
            df=analysis.result.df,
            j_per_df=analysis.result.j_per_df,
            expected_band_low=analysis.result.expected_band_low,
            expected_band_high=analysis.result.expected_band_high,
            tail_fraction=analysis.result.tail_fraction,
        )
    finally:
        tab.save()


DEFINITION = chain_t1
