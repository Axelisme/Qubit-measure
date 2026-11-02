"""Measure ground-heralded Rabi oscillations versus drive gain.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

from typing import Annotated

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2.twotone.rabi.amp_rabi.gui import AmplitudeRabiSummary


class HeraldingAmpRabiInputs(RecipeInputs):
    """Inputs for heralded amplitude Rabi; all units and null policies are declared per
    field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse-readout library key including classifier. Sources are chosen by the Form.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Drive library key. No pi calibration is required for this sweep.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Reset module option.",
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
        description="Acquisition repetition count.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count.",
    )

    skip: int | None | Keep = Field(
        KEEP,
        description="Skip points. Analysis only; omitted keeps value, null passes to owner.",
    )


class HeraldingAmpRabiReport(RecipeReport):
    """Amplitude Rabi results and failure reason; cancellation returns no report."""

    pi_gain: float | None = Field(
        description="Pi rotation gain in native amplitude units; null means not estimable."
    )
    pi_gain_err: float | None = Field(
        description="Uncertainty of pi rotation gain; null means not estimable."
    )
    pi2_gain: float | None = Field(
        description="Pi/2 rotation gain in native amplitude units; null means not estimable."
    )
    pi2_gain_err: float | None = Field(
        description="Uncertainty of pi/2 rotation gain; null means not estimable."
    )


@recipe(
    "heralding_amp_rabi",
    description="Run ground-heralded gain Rabi and save per-class shot sums and analysis.",
)
async def heralding_amp_rabi(
    session: RecipeSession, inputs: HeraldingAmpRabiInputs
) -> HeraldingAmpRabiReport:
    """Sweep drive gain at fixed pulse length to calibrate pi and pi/2 rotations.

    Store per-class shot sums; compute herald-g conditional IQ for fitting.
    Readout accepts circle, linear or centroid3. Herald states are g/e/other
    for circle (other is rejection), g/e for linear (no rejection), and g/e/f
    for centroid3 (f is not other). Circle requires a calibrated GE Post radius.
    Only shots classified as herald-g contribute to signal IQ and fitting.
    After optional reset, herald ground with the saved readout classifier,
    apply the drive pulse and read out at each gain. Average only accepted shots.
    No active reset or joint herald is added. Fit the Rabi oscillations to estimate pi and pi/2 gains,
    and ask before applying the proposed calibration; skipping writes nothing.

    points → expts sets the sweep sample count.
    Analysis controls: skip. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "heralding/amp_rabi",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
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
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=AmplitudeRabiSummary,
            updates={"skip": inputs.skip},
        )
        if analysis.result is None:
            return HeraldingAmpRabiReport(
                pi_gain=None,
                pi_gain_err=None,
                pi2_gain=None,
                pi2_gain_err=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return HeraldingAmpRabiReport(
            pi_gain=report_number(analysis.result.pi_amp),
            pi_gain_err=report_number(analysis.result.pi_amp_err),
            pi2_gain=report_number(analysis.result.pi2_amp),
            pi2_gain_err=report_number(analysis.result.pi2_amp_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = heralding_amp_rabi
