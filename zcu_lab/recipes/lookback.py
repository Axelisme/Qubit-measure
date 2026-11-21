"""Readout arrival-time trace and proposed ADC trigger timing."""

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

from zcu_lab.v2.lookback.gui import LookbackSummary


class LookbackInputs(RecipeInputs):
    """Inputs for lookback; all units and null policies are declared per field."""

    target: SingleTarget = Field(
        default_factory=lambda: KEEP,
        description="Open/Defaults for this qubit and its resonator. Omit or use null to keep the current aliases.",
    )
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Select a library module in Readout. Omit to retain the selection; null is not accepted.",
    )
    frequency_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Readout › Frequency (MHz), the shared RF drive and demodulation frequency.",
    )
    readout_length_us: float | None | Keep = Field(
        KEEP,
        description="Set Readout › RO length (µs), the ADC capture duration.",
    )
    trigger_offset_us: float | None | Keep = Field(
        KEEP,
        description="Set Readout › Trig offset (µs), the ADC window start relative to readout start.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Averaging rounds (count), the number of one-shot traces to average.",
    )

    ratio: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Arrival threshold / peak (a.u.), a fraction of the peak used to find the rising edge.",
    )
    smooth: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Gaussian smoothing width (samples), the Gaussian sigma applied to I and Q.",
    )
    plot_fit: bool | None | Keep = Field(
        KEEP,
        description="Set Analyze › Show arrival markers. This changes the display, not the arrival estimate.",
    )


class LookbackReport(RecipeReport):
    """Lookback results and failure reason; cancellation returns no report."""

    trigger_offset_us: float | None = Field(
        description="Estimated readout arrival (µs); null means analysis returned no value."
    )


@recipe(
    "lookback",
    description="Capture the time-resolved ADC readout trace, estimate its arrival, ask before saving that time to the resonator, and save the run.",
)
async def lookback(session: RecipeSession, inputs: LookbackInputs) -> LookbackReport:
    """Capture the readout pulse's arrival and ADC time envelope.

    Read out at a fixed frequency without reset or initialization pulses.
    Analyze estimates readout arrival from the rising magnitude envelope.
    Ask before applying the timing proposal, then save raw trace and analysis.

    No n → expts input: acquisition uses the readout duration and rounds.
    Analysis controls: ratio, smooth, plot_fit. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "lookback",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    if (
        not isinstance(inputs.frequency_mhz, Keep)
        or not isinstance(inputs.readout_length_us, Keep)
        or not isinstance(inputs.trigger_offset_us, Keep)
    ):
        tab.command("readout", "to_inline")
    tab.set("readout.freq", inputs.frequency_mhz)
    tab.set("readout.ro_length", inputs.readout_length_us)
    tab.set("readout.trig_offset", inputs.trigger_offset_us)
    tab.set("rounds", inputs.rounds)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=LookbackSummary,
            updates={
                "ratio": inputs.ratio,
                "smooth": inputs.smooth,
                "plot_fit": inputs.plot_fit,
            },
        )
        if analysis.result is None:
            return LookbackReport(
                trigger_offset_us=None,
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)

        return LookbackReport(
            trigger_offset_us=report_number(analysis.result.predict_offset),
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = lookback
