"""Measure Rabi oscillations versus drive gain to calibrate rotation amplitudes.

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


class AmplitudeRabiInputs(RecipeInputs):
    """Inputs for amplitude rabi; all units and null policies are declared per
    field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value; null clears/disables it. Sources are chosen by the Form.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Drive library key; omission retains the Form value; null clears/disables it. No pi calibration is required for this sweep.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Reset module option; omission retains the Form selection, null disables it.",
    )
    frequency_mhz: float | None | Keep = Field(
        KEEP,
        description="Fixed pulse frequency in MHz. Omission retains the Form value; null clears it.",
    )
    pulse_length_us: float | None | Keep = Field(
        KEEP,
        description="Fixed drive pulse duration in us; omission retains the Form value; null clears/disables it.",
    )
    gain_range: (
        Annotated[list[float], Field(min_length=2, max_length=2)] | None | Keep
    ) = Field(
        KEEP,
        description="Two native gain endpoints; omission retains the Form value; null clears/disables it.",
    )
    points: int | None | Keep = Field(
        KEEP,
        description="Sweep sample count; omission retains the Form value; null clears/disables it.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Acquisition repetition count; omission retains the Form value; null clears/disables it.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count; omission retains the Form value; null clears/disables it.",
    )

    skip: int | None | Keep = Field(
        KEEP,
        description="Skip points; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class AmplitudeRabiReport(RecipeReport):
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
    "amplitude_rabi",
    description="Run one gain Rabi sweep and save raw data and Primary analysis.",
)
async def amplitude_rabi(
    session: RecipeSession, inputs: AmplitudeRabiInputs
) -> AmplitudeRabiReport:
    """Sweep drive gain at fixed pulse length to calibrate pi and pi/2 rotations.

    After optional reset, apply the drive pulse and read out the qubit at each
    gain. Fit the Rabi oscillations to estimate pi and pi/2 gains,
    and ask before applying the proposed calibration; skipping writes nothing.

    points → expts sets the sweep sample count.
    Analysis controls: skip. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/rabi/amp_rabi",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("modules.reset", inputs.use_reset)
    tab.select("modules.readout", inputs.readout_ref)
    tab.select("modules.qub_pulse", inputs.drive_ref)
    if not isinstance(inputs.frequency_mhz, Keep) or not isinstance(
        inputs.pulse_length_us, Keep
    ):
        pulse_selector = tab.read(("modules", "qub_pulse")).selector
        if pulse_selector is not None and pulse_selector.mode == "ref":
            tab.command("modules.qub_pulse", "to_inline")
    if not isinstance(inputs.pulse_length_us, Keep):
        waveform_selector = tab.read(("modules", "qub_pulse", "waveform")).selector
        if waveform_selector is not None and waveform_selector.mode == "ref":
            tab.command("modules.qub_pulse.waveform", "to_inline")
    tab.set("modules.qub_pulse.freq", inputs.frequency_mhz)
    tab.set("modules.qub_pulse.waveform.length", inputs.pulse_length_us)
    if inputs.gain_range is None:
        tab.set("sweep.gain.start", None)
        tab.set("sweep.gain.stop", None)
    elif not isinstance(inputs.gain_range, Keep):
        tab.set("sweep.gain.start", inputs.gain_range[0])
        tab.set("sweep.gain.stop", inputs.gain_range[1])
    tab.set("sweep.gain.n", inputs.points)
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
            return AmplitudeRabiReport(
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

        return AmplitudeRabiReport(
            pi_gain=report_number(analysis.result.pi_gain),
            pi_gain_err=report_number(analysis.result.pi_gain_err),
            pi2_gain=report_number(analysis.result.pi2_gain),
            pi2_gain_err=report_number(analysis.result.pi2_gain_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = amplitude_rabi
