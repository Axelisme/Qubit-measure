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
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Drive library key with a const or flat_top waveform.",
    )
    frequency_mhz: float | None | Keep = Field(
        KEEP,
        description="Fixed pulse frequency in MHz.",
    )
    gain: float | None | Keep = Field(
        KEEP,
        description="Fixed pulse amplitude in native gain units.",
    )
    max_length_us: float | None | Keep = Field(
        KEEP,
        description="Pulse-length sweep stop in us.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Sweep sample count.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Cycles per length in each round, at least 66.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Round count; the live plot updates after each round.",
    )


class ChainLenRabiReport(RecipeReport):
    """Chain Rabi fit and its checks; cancellation returns no report."""

    pi_length_us: float | None = Field(
        description="Pi length from the chain fit in us; null means not estimable."
    )
    pi_length_err_us: float | None = Field(
        description="Pi length uncertainty in us; null means not estimable."
    )
    rabi_frequency_mhz: float | None = Field(
        description="Rabi frequency in MHz; null means not estimable."
    )
    rabi_frequency_err_mhz: float | None = Field(
        description="Rabi frequency uncertainty in MHz; null means not estimable."
    )
    reliability: float | None = Field(
        description="Diagnostic R of the frequency error, no threshold."
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
    first_read_pi_length_us: float | None = Field(
        description="Pi length in us of the first-read twotone fit; null means not estimable."
    )


@recipe(
    "chain_len_rabi",
    description="Run one K-cycle readout-chain length Rabi, fit all 65 numbers, save sums and analysis.",
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
