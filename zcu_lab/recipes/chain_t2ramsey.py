"""T2 Ramsey with a K-cycle readout chain at every delay.

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
from zcu_lab.v2.chain.t2ramsey.gui import ChainT2RamseySummary


class ChainT2RamseyInputs(RecipeInputs):
    """Inputs for chain T2 Ramsey; units and null policies are per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Set Readout to a library key for the single-tone pulse readout played twice per cycle block.",
    )
    pi2_ref: str | None | Keep = Field(
        KEEP,
        description="Set π/2 pulse to the calibrated library pulse.",
    )
    max_delay_us: float | None | Keep = Field(
        KEEP,
        description="Set Free-evolution wait › Stop (µs), the last wait.",
    )
    detune_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Ramsey phase-ramp detune (MHz) for the second π/2 phase ramp; the RF frequency is unchanged.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Set Free-evolution wait › Expts (count), the number of sweep points.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), the number of cycles at every sweep point in one round; at least 66.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), the planned number of passes through the sweep.",
    )


class ChainT2RamseyReport(RecipeReport, JointResidualSummary):
    """Joint Ramsey estimates and scalar residuals; cancellation returns no report."""

    t2_us: float | None = Field(description="T2* (µs); null if unavailable.")
    t2_err_us: float | None = Field(
        description="T2* Wald standard error (µs); null if unavailable."
    )
    detuning_mhz: float | None = Field(
        description="Fringe frequency (MHz), including the phase ramp; null if unavailable."
    )
    detuning_err_mhz: float | None = Field(
        description="Fringe frequency Wald standard error (MHz); null if unavailable."
    )
    tf_us: float | None = Field(
        description="Third-state decay time (µs); null if unavailable."
    )
    tf_err_us: float | None = Field(
        description="Third-state decay time Wald standard error (µs); null if unavailable."
    )
    reliability: float | None = Field(
        description="R, the inverse phase uncertainty from the frequency error over the sweep (no threshold); null if unavailable."
    )
    first_read_t2_us: float | None = Field(
        description="First-read T2* (µs); null if unavailable."
    )


@recipe(
    "chain_t2ramsey",
    description="Sweep K-cycle Ramsey wait, fit all 65 numbers jointly, and save the sums, estimates and residual diagnostics without writeback.",
)
async def chain_t2ramsey(
    session: RecipeSession, inputs: ChainT2RamseyInputs
) -> ChainT2RamseyReport:
    """Sweep the Ramsey wait with one K-cycle per rep: K, then (pi/2, wait, pi/2, read, read) x 8.

    The second pi/2 has opposite phase plus the detune ramp. relax_delay is
    the thermalisation wait K; the Form starts it at 5 T1 and overrides can
    change it. Primary fits the first read with the twotone Ramsey fit for
    the starting values, then fits all 65 numbers jointly. It reports T2,
    the fringe frequency and the third-state decay time with Wald errors. It
    also reports R, J, df, J/df, the chi-square reference band endpoints and
    the fraction of whitened components with |r| > 3. These numbers do not
    decide acceptance. The per-delay residuals stay in the figure. Save
    keeps the running sums and the analysis. No writeback is proposed.

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
                J=None,
                df=None,
                j_per_df=None,
                expected_band_low=None,
                expected_band_high=None,
                tail_fraction=None,
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


DEFINITION = chain_t2ramsey
