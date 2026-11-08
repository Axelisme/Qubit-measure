"""Qubit drive-frequency sweep with a readout chain at every point.

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

from zcu_lab.v2.twotone.chain.gui import TwotoneChainSummary


class TwotoneChainInputs(RecipeInputs):
    """Inputs for the readout-chain sweep; units and null policies are per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Pulse readout library key. Sources are chosen by the Form.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Drive library key. No pi calibration is required for this sweep.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Reset module option.",
    )
    center_mhz: float = Field(description="Sweep center in MHz.")
    span_mhz: float = Field(
        gt=0,
        description="Full frequency span in MHz; endpoints are center minus/plus span/2.",
    )
    gain: float | None | Keep = Field(
        KEEP,
        description="Fixed drive amplitude in native gain units.",
    )
    pulse_length_us: float | None | Keep = Field(
        KEEP,
        description="Fixed drive pulse duration in us.",
    )
    points: int = Field(
        ge=2,
        description="Sweep sample count.",
    )
    reps: int | None | Keep = Field(
        KEEP,
        description="Fast sweeps per round, at least 2.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Round count; the live curve updates after each round.",
    )

    expected_center_mhz: float | None | Keep = Field(
        KEEP,
        description="Analysis only: fit the peak nearest this MHz; null fits the strongest.",
    )


class TwotoneChainReport(RecipeReport):
    """Sigma-peak centre and width; cancellation returns no report."""

    frequency_mhz: float | None = Field(
        description="Fitted sigma-peak centre in MHz; null means not estimable."
    )
    frequency_err_mhz: float | None = Field(
        description="Centre uncertainty in MHz; null means not estimable."
    )
    width_mhz: float | None = Field(
        description="Sigma-peak FWHM in MHz, an approximate line width; null means not estimable."
    )
    width_err_mhz: float | None = Field(
        description="Width uncertainty in MHz; null means not estimable."
    )
    level_sigma: float | None = Field(
        description="5% false-mark level in sigma; null means analysis gave no result."
    )


@recipe(
    "twotone_chain",
    description="Run one readout-chain two-tone sweep, fit the sigma peak, save sums and analysis.",
)
async def twotone_chain(
    session: RecipeSession, inputs: TwotoneChainInputs
) -> TwotoneChainReport:
    """Sweep qubit drive frequency with 48 (drive, readout) cycles per point.

    Each point applies optional reset, then 48 cycles of the fixed-length/gain
    drive and a readout without reset. Primary recomputes the distance from
    background in sigma and fits one peak for its centre and approximate
    width. No writeback is proposed or requested. Save keeps the running sums
    and the analysis.

    points → expts sets the sweep sample count.
    expected_center_mhz selects the peak to fit; null fits the most prominent.
    Omission retains the analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "twotone/chain",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("reset", inputs.use_reset)
    tab.select("qub_pulse", inputs.drive_ref)
    if not isinstance(inputs.gain, Keep) or not isinstance(
        inputs.pulse_length_us, Keep
    ):
        tab.command("qub_pulse", "to_inline")
    if not isinstance(inputs.pulse_length_us, Keep):
        tab.command("qub_pulse.waveform", "to_inline")
    tab.set("freq.start", inputs.center_mhz - inputs.span_mhz / 2)
    tab.set("freq.stop", inputs.center_mhz + inputs.span_mhz / 2)
    tab.set("freq.n", inputs.points)
    tab.set("qub_pulse.gain", inputs.gain)
    tab.set("qub_pulse.waveform.length", inputs.pulse_length_us)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=TwotoneChainSummary,
            updates={"expected_center_mhz": inputs.expected_center_mhz},
        )
        if analysis.result is None:
            return TwotoneChainReport(
                frequency_mhz=None,
                frequency_err_mhz=None,
                width_mhz=None,
                width_err_mhz=None,
                level_sigma=None,
                analysis_failure=analysis.reason,
            )
        return TwotoneChainReport(
            frequency_mhz=report_number(analysis.result.freq),
            frequency_err_mhz=report_number(analysis.result.freq_err),
            width_mhz=report_number(analysis.result.fwhm),
            width_err_mhz=report_number(analysis.result.fwhm_err),
            level_sigma=report_number(analysis.result.level),
        )
    finally:
        tab.save()


DEFINITION = twotone_chain
