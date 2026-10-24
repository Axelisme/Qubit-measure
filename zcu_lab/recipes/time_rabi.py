"""Measure Rabi oscillations versus pulse length to calibrate rotation durations.

Recipe points → expts: scan sample counts map to the GUI sweep Expts field.
"""

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

from zcu_lab.v2.twotone.rabi.len_rabi.gui import TimeRabiSummary


class TimeRabiInputs(RecipeInputs):
    """Inputs for time rabi; all units and null policies are declared per field."""

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
    gain: float | None | Keep = Field(
        KEEP,
        description="Fixed pulse amplitude in native gain units; omission retains the Form value; null clears/disables it.",
    )
    max_length_us: float | None | Keep = Field(
        KEEP,
        description="Pulse-length sweep stop in us; omission retains the Form value; null clears/disables it.",
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

    decay: bool | None | Keep = Field(
        KEEP,
        description="Fit decay envelope; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    fit_phase: bool | None | Keep = Field(
        KEEP,
        description="Fit phase offset; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class TimeRabiReport(RecipeReport):
    """Time Rabi results and failure reason; cancellation returns no report."""

    pi_length_us: float | None = Field(
        description="Pi pulse duration in us; null means not estimable."
    )
    pi_length_err_us: float | None = Field(
        description="Pi duration uncertainty in us; null means not estimable."
    )
    pi2_length_us: float | None = Field(
        description="Pi/2 pulse duration in us; null means not estimable."
    )
    pi2_length_err_us: float | None = Field(
        description="Pi/2 duration uncertainty in us; null means not estimable."
    )
    rabi_frequency_mhz: float | None = Field(
        description="Fitted Rabi oscillation frequency in MHz; null means not estimable."
    )
    rabi_frequency_err_mhz: float | None = Field(
        description="Rabi frequency uncertainty in MHz; null means not estimable."
    )


@recipe(
    "time_rabi",
    description="Run one length Rabi sweep and save raw data and Primary analysis.",
)
async def time_rabi(session: RecipeSession, inputs: TimeRabiInputs) -> TimeRabiReport:
    """Sweep pulse length at fixed gain to calibrate pi and pi/2 rotations.

    After optional reset, apply the drive pulse and read out the qubit at each
    length. Fit the Rabi oscillations to estimate pi and pi/2
    durations, and ask before applying the proposed calibration; skipping writes
    nothing.

    points → expts sets the sweep sample count.
    Analysis controls: decay, fit_phase. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/rabi/len_rabi",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("modules.reset", inputs.use_reset)
    tab.select("modules.readout", inputs.readout_ref)
    tab.select("modules.qub_pulse", inputs.drive_ref)
    if not isinstance(inputs.frequency_mhz, Keep) or not isinstance(inputs.gain, Keep):
        tab.command("modules.qub_pulse", "to_inline")
    tab.set("modules.qub_pulse.freq", inputs.frequency_mhz)
    tab.set("modules.qub_pulse.gain", inputs.gain)
    tab.set("sweep.length.stop", inputs.max_length_us)
    tab.set("sweep.length.n", inputs.points)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=TimeRabiSummary,
            updates={
                "decay": inputs.decay,
                "fit_phase": inputs.fit_phase,
            },
        )
        if analysis.result is None:
            return TimeRabiReport(
                pi_length_us=None,
                pi_length_err_us=None,
                pi2_length_us=None,
                pi2_length_err_us=None,
                rabi_frequency_mhz=None,
                rabi_frequency_err_mhz=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return TimeRabiReport(
            pi_length_us=report_number(analysis.result.pi_len),
            pi_length_err_us=report_number(analysis.result.pi_len_err),
            pi2_length_us=report_number(analysis.result.pi2_len),
            pi2_length_err_us=report_number(analysis.result.pi2_len_err),
            rabi_frequency_mhz=report_number(analysis.result.rabi_f),
            rabi_frequency_err_mhz=report_number(analysis.result.rabi_f_err),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = time_rabi
