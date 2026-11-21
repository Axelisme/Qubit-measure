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
        description="Set Readout to a library key for the single-tone pulse readout played twice per cycle block.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Set Drive pulse to a library key; the sweep replaces gain, so π gain need not be calibrated.",
    )
    frequency_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Drive pulse › Frequency (MHz), the fixed RF carrier; the pulse switches to inline editing.",
    )
    pulse_length_us: float | None | Keep = Field(
        KEEP,
        description="Set Drive pulse › Waveform › Length (µs), the fixed drive duration; switches to inline editing.",
    )
    gain_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP,
        description="Set Drive gain › Start and Stop (a.u.), in that order.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Set Drive gain › Expts (count), the number of sweep points.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Set Reps (count), the number of cycles at every sweep point in one round; at least 66.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), the planned number of passes through the sweep.",
    )


class ChainAmpRabiReport(RecipeReport):
    """Chain Rabi fit and its checks; cancellation returns no report."""

    pi_gain: float | None = Field(description="π gain (a.u.); null if unavailable.")
    pi_gain_err: float | None = Field(
        description="π gain Wald standard error (a.u.); null if unavailable."
    )
    frequency_per_gain: float | None = Field(
        description="Rabi frequency (cycles per gain unit); null if unavailable."
    )
    frequency_per_gain_err: float | None = Field(
        description="Rabi frequency Wald standard error (cycles per gain unit); null if unavailable."
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
    first_read_pi_gain: float | None = Field(
        description="First-read π gain (a.u.); null if unavailable."
    )


@recipe(
    "chain_amp_rabi",
    description="Sweep drive gain in K-cycle amplitude Rabi; use the first-read fit to seed a joint three-state fit, then save without writeback.",
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
