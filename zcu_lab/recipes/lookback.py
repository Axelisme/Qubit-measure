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
    """Lookback Run and Analyze inputs; see the guide for null handling."""

    target: SingleTarget = Field(
        default_factory=lambda: KEEP,
        description="Open/Defaults for this qubit and its resonator. Omit or use null to keep the current aliases.",
    )
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Select a Readout module first. Omit to keep it. Schema allows null; required Readout rejects it.",
    )
    frequency_mhz: float | None | Keep = Field(
        KEEP,
        description="Readout › Frequency (MHz). Copies a library Readout to Inline before editing; no library writeback.",
    )
    readout_length_us: float | None | Keep = Field(
        KEEP,
        description="Readout › RO length (µs). Copies a library Readout to Inline before editing; no library writeback.",
    )
    trigger_offset_us: float | None | Keep = Field(
        KEEP,
        description="Readout › Trig offset (µs). Copies a library Readout to Inline before editing; no library writeback.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Set Averaging rounds (count), the number of one-shot traces to average.",
    )

    ratio: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Arrival threshold / peak (a.u.). Omit to keep it; Analyze rejects null.",
    )
    smooth: float | None | Keep = Field(
        KEEP,
        description="Set Analyze › Gaussian smoothing width (samples). Omit to keep it; Analyze rejects null.",
    )
    plot_fit: bool | None | Keep = Field(
        KEEP,
        description="Set Analyze › Show arrival markers only. Omit to keep it; Analyze rejects null.",
    )


class LookbackReport(RecipeReport):
    """Lookback results and failure reason; cancellation returns no report."""

    trigger_offset_us: float | None = Field(
        description="Estimated readout arrival (µs); null means analysis returned no value."
    )


@recipe(
    "lookback",
    description="Capture the ADC trace and estimate arrival. Analyze rejects null after Run, before analysis starts. The recipe still attempts to save the trace. Confirm before writing arrival to the resonator.",
)
async def lookback(session: RecipeSession, inputs: LookbackInputs) -> LookbackReport:
    """Capture the readout pulse's arrival and ADC time envelope.

    Read out at a fixed frequency without reset or initialization pulses.
    Analyze estimates readout arrival from the rising magnitude envelope.
    Ask before applying any timing proposal. After Run, attempt Save even if
    Analyze rejects its inputs.

    Named Readout field edits copy the selected library Readout to Inline first.
    Omit Analyze inputs to retain their values. Analyze validates and rejects
    null updates after Run and before starting analysis.
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
