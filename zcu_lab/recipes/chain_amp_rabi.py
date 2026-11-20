"""Amplitude Rabi with a K-cycle readout chain at every gain.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Annotated

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

from zcu_lab.v2.chain.amp_rabi.gui import ChainAmpRabiSummary


class ChainAmpRabiInputs(RecipeInputs):
    """Inputs for chain amplitude Rabi; units and null policies are per field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Drive library key. No pi calibration is required for this sweep.",
    )
    frequency_mhz: float | None | Keep = Field(
        KEEP,
        description="Fixed pulse frequency in MHz.",
    )
    pulse_length_us: float | None | Keep = Field(
        KEEP,
        description="Fixed drive pulse duration in us.",
    )
    gain_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP,
        description="Two native gain endpoints.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Sweep sample count.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Cycles per gain in each round, at least 66.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Round count; the live plot updates after each round.",
    )


class ChainAmpRabiReport(RecipeReport):
    """Chain Rabi fit and its checks; cancellation returns no report."""

    pi_gain: float | None = Field(
        description="Pi gain from the chain fit in native units; null means not estimable."
    )
    pi_gain_err: float | None = Field(
        description="Pi gain uncertainty; null means not estimable."
    )
    frequency_per_gain: float | None = Field(
        description="Rabi frequency in cycles per gain unit; null means not estimable."
    )
    frequency_per_gain_err: float | None = Field(
        description="Rabi frequency uncertainty; null means not estimable."
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
    first_read_pi_gain: float | None = Field(
        description="Pi gain of the first-read twotone fit; null means not estimable."
    )


@recipe(
    "chain_amp_rabi",
    description="Run one K-cycle readout-chain gain Rabi, fit all 65 numbers, save sums and analysis.",
)
async def chain_amp_rabi(
    session: RecipeSession, inputs: ChainAmpRabiInputs
) -> ChainAmpRabiReport:
    """Sweep drive gain with one K-cycle per rep: K, then (pulse, read, read) x 8.

    relax_delay is the thermalisation wait K; the Form starts it at 5 T1 and
    overrides can change it. Primary fits the first read with the twotone Rabi
    fit for the starting value, then fits all 65 numbers jointly. It reports
    the pi gain and Rabi frequency with Wald errors, R and three checks. No
    writeback is proposed or requested. Save keeps the running sums and the
    analysis.

    points → expts sets the sweep sample count.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "chain/amp_rabi",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("qub_pulse", inputs.drive_ref)
    if not isinstance(inputs.frequency_mhz, Keep) or not isinstance(
        inputs.pulse_length_us, Keep
    ):
        tab.command("qub_pulse", "to_inline")
    if not isinstance(inputs.pulse_length_us, Keep):
        tab.command("qub_pulse.waveform", "to_inline")
    tab.set("qub_pulse.freq", inputs.frequency_mhz)
    tab.set("qub_pulse.waveform.length", inputs.pulse_length_us)
    if inputs.gain_range is None:
        tab.set("gain.start", None)
        tab.set("gain.stop", None)
    elif not isinstance(inputs.gain_range, Keep):
        tab.set("gain.start", inputs.gain_range[0])
        tab.set("gain.stop", inputs.gain_range[1])
    tab.set("gain.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=ChainAmpRabiSummary)
        if analysis.result is None:
            return ChainAmpRabiReport(
                pi_gain=None,
                pi_gain_err=None,
                frequency_per_gain=None,
                frequency_per_gain_err=None,
                reliability=None,
                within_point=None,
                sharing=None,
                curve=None,
                first_read_pi_gain=None,
                analysis_failure=analysis.reason,
            )
        return ChainAmpRabiReport(
            pi_gain=report_number(analysis.result.pi_amp),
            pi_gain_err=report_number(analysis.result.pi_amp_err),
            frequency_per_gain=report_number(analysis.result.f),
            frequency_per_gain_err=report_number(analysis.result.f_err),
            reliability=report_number(analysis.result.reliability),
            within_point=analysis.result.within_point,
            sharing=analysis.result.sharing,
            curve=analysis.result.curve,
            first_read_pi_gain=report_number(analysis.result.first_read_pi_amp),
        )
    finally:
        tab.save()


DEFINITION = chain_amp_rabi
