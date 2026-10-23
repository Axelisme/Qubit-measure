"""Readout arrival-time trace and proposed ADC trigger timing."""

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

from zcu_lab.v2.lookback.gui import LookbackSummary


class LookbackInputs(RecipeInputs):
    """Inputs for lookback; all units and null policies are declared per field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value; null clears/disables it. Sources are chosen by the Form.",
    )
    frequency_mhz: float | None | Keep = Field(
        KEEP,
        description="Readout frequency in MHz; omission retains the Form value, null clears it.",
    )
    readout_length_us: float | None | Keep = Field(
        KEEP,
        description="ADC acquisition window in us; omission retains the Form value; null clears/disables it.",
    )
    trigger_offset_us: float | None | Keep = Field(
        KEEP,
        description="ADC trigger offset in us; omission retains the Form value; null clears/disables it.",
    )
    rounds: int | None | Keep = Field(
        KEEP,
        description="Averaging round count; omission retains the Form value; null clears/disables it.",
    )

    ratio: float | None | Keep = Field(
        KEEP,
        description="Threshold ratio; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    smooth: float | None | Keep = Field(
        KEEP,
        description="Smooth sigma; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    plot_fit: bool | None | Keep = Field(
        KEEP,
        description="Plot fit; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )


class LookbackReport(RecipeReport):
    """Lookback results and failure reason; cancellation returns no report."""

    trigger_offset_us: float | None = Field(
        description="Proposed ADC trigger offset in us; null means not estimable."
    )


@recipe(
    "lookback",
    description="Capture readout arrival timing in lookback; analyze, ask before writeback, and save.",
)
async def lookback(session: RecipeSession, inputs: LookbackInputs) -> LookbackReport:
    """Capture the readout pulse's arrival and ADC time envelope.

    Read out at a fixed frequency without reset or initialization pulses.
    Primary estimates the trigger offset from the rising signal envelope.
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
    tab.select("modules.readout", inputs.readout_ref)
    if (
        not isinstance(inputs.frequency_mhz, Keep)
        or not isinstance(inputs.readout_length_us, Keep)
        or not isinstance(inputs.trigger_offset_us, Keep)
    ):
        pulse_selector = tab.read(("modules", "readout")).selector
        if pulse_selector is not None and pulse_selector.mode == "ref":
            tab.command("modules.readout", "to_inline")
    tab.set("modules.readout.freq", inputs.frequency_mhz)
    tab.set("modules.readout.ro_length", inputs.readout_length_us)
    tab.set("modules.readout.trig_offset", inputs.trigger_offset_us)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
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
