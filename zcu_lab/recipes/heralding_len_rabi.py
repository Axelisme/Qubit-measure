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


class HeraldingLenRabiInputs(RecipeInputs):
    """Inputs for time rabi; all units and null policies are declared per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse-readout library key with calibrated classifier. Null passes to the Form owner.",
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
        description="Acquisition repetition count.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count.",
    )

    decay: bool | None | Keep = Field(
        KEEP,
        description="Fit decay envelope. Analysis only; omitted keeps value, null passes to owner.",
    )
    fit_phase: bool | None | Keep = Field(
        KEEP,
        description="Fit phase offset. Analysis only; omitted keeps value, null passes to owner.",
    )


class HeraldingLenRabiReport(RecipeReport):
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
    "heralding_len_rabi",
    description="Run ground-heralded length Rabi; save per-class shot sums and analysis.",
)
async def heralding_len_rabi(
    session: RecipeSession, inputs: HeraldingLenRabiInputs
) -> HeraldingLenRabiReport:
    """Sweep pulse length at fixed gain to calibrate pi and pi/2 rotations.

    Store per-class shot sums; compute herald-g conditional IQ for fitting.
    Readout accepts circle, linear or centroid3. Herald states are g/e/other
    for circle (other is rejection), g/e for linear (no rejection), and g/e/f
    for centroid3 (f is not other). Circle requires a calibrated GE Post radius.
    Only shots classified as herald-g contribute to signal IQ and fitting.
    After optional reset, herald ground with the saved readout classifier, apply
    optional initialization and the drive, then read accepted signal shots at each
    length. Fit the Rabi oscillations to estimate pi and pi/2
    durations, and ask before applying the proposed calibration; skipping writes
    nothing. This is host post-selection, not active reset. Circle heralding
    requires a calibrated radius. Pass fraction is not independent ground purity.

    points → expts sets the sweep sample count.
    Analysis controls: decay, fit_phase. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "heralding/len_rabi",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
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
            return HeraldingLenRabiReport(
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

        return HeraldingLenRabiReport(
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


DEFINITION = heralding_len_rabi
