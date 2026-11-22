"""Length Rabi with a K-cycle readout chain at every pulse length.

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

from zcu_lab.v2.chain.len_rabi.gui import ChainLenRabiSummary


class ChainLenRabiInputs(RecipeInputs):
    """Inputs for chain length Rabi; units and null policies are per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Set Readout to a library key for the single-tone pulse readout played twice per cycle block.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Set Drive pulse to a library pulse with a constant or flat-top waveform.",
    )
    frequency_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Drive pulse › Frequency (MHz), the fixed RF carrier; the pulse switches to inline editing.",
    )
    gain: float | None | Keep = Field(
        KEEP,
        description="Set Drive pulse › Gain (a.u.), the fixed signed drive gain.",
    )
    max_length_us: float | None | Keep = Field(
        KEEP,
        description="Set Drive duration › Stop (µs), the last drive duration.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Set Drive duration › Expts (count), the number of sweep points.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), the number of cycles at every sweep point in one round; at least 66.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), the planned number of passes through the sweep.",
    )


class ChainLenRabiReport(RecipeReport):
    """Chain Rabi fit and its checks; cancellation returns no report."""

    pi_length_us: float | None = Field(
        description="π duration (µs); null if unavailable."
    )
    pi_length_err_us: float | None = Field(
        description="π duration Wald standard error (µs); null if unavailable."
    )
    rabi_frequency_mhz: float | None = Field(
        description="Rabi frequency (MHz); null if unavailable."
    )
    rabi_frequency_err_mhz: float | None = Field(
        description="Rabi frequency Wald standard error (MHz); null if unavailable."
    )
    reliability: float | None = Field(
        description="R, the inverse phase uncertainty from the frequency error over the sweep (no threshold); null if unavailable."
    )
    within_point: CycleCheck | None = Field(
        description="Within-point check J, df, p; null without analysis."
    )
    sharing: CycleCheck | None = Field(
        description="Sharing check J, df, p; null without analysis."
    )
    curve: CycleCheck | None = Field(
        description="Curve check J, df, p; null without analysis."
    )
    first_read_pi_length_us: float | None = Field(
        description="First-read π duration (µs); null if unavailable."
    )


@recipe(
    "chain_len_rabi",
    description="Sweep drive duration in K-cycle length Rabi; use the first-read fit to seed a joint three-state fit, then save without writeback.",
)
async def chain_len_rabi(
    session: RecipeSession, inputs: ChainLenRabiInputs
) -> ChainLenRabiReport:
    """Sweep pulse length with one K-cycle per rep: K, then (pulse, read, read) x 8.

    The FPGA sweeps the length, so the drive must be const or flat_top.
    relax_delay is the thermalisation wait K; the Form starts it at 5 T1 and
    overrides can change it. Primary fits the first read with the twotone Rabi
    fit for the starting value, then fits all 65 numbers jointly. It reports
    the pi length and Rabi frequency with Wald errors, R and three checks. No
    writeback is proposed or requested. Save keeps the running sums and the
    analysis.

    points → expts sets the sweep sample count.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "chain/len_rabi",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("qub_pulse", inputs.drive_ref)
    if not isinstance(inputs.frequency_mhz, Keep) or not isinstance(inputs.gain, Keep):
        tab.command("qub_pulse", "to_inline")
    tab.set("qub_pulse.freq", inputs.frequency_mhz)
    tab.set("qub_pulse.gain", inputs.gain)
    tab.set("length.stop", inputs.max_length_us)
    tab.set("length.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ChainLenRabiSummary)
        if analysis.result is None:
            return ChainLenRabiReport(
                pi_length_us=None,
                pi_length_err_us=None,
                rabi_frequency_mhz=None,
                rabi_frequency_err_mhz=None,
                reliability=None,
                within_point=None,
                sharing=None,
                curve=None,
                first_read_pi_length_us=None,
                analysis_failure=analysis.reason,
            )
        return ChainLenRabiReport(
            pi_length_us=report_number(analysis.result.pi_len),
            pi_length_err_us=report_number(analysis.result.pi_len_err),
            rabi_frequency_mhz=report_number(analysis.result.f),
            rabi_frequency_err_mhz=report_number(analysis.result.f_err),
            reliability=report_number(analysis.result.reliability),
            within_point=analysis.result.within_point,
            sharing=analysis.result.sharing,
            curve=analysis.result.curve,
            first_read_pi_length_us=report_number(analysis.result.first_read_pi_len),
        )
    finally:
        tab.save()


DEFINITION = chain_len_rabi
