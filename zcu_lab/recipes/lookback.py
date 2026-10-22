"""Readout arrival-time trace and proposed ADC trigger timing."""

from typing import Literal

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import (
    RecipeSession,
    RecipeTab,
    recipe,
)
from zcu_tools.mcp.measure.recipe_contracts import RecipeNeedsParameters

from .helpers import (
    FrequencyCalibration,
    apply_frequency,
    choose_frequency,
    report_number,
    retain_scalar,
    select_or_disable,
)


def _set_readout_timing(
    tab: RecipeTab, readout_length_us: float | None, trigger_offset_us: float | None
) -> None:
    """Keep omitted key values and provenance; convert only for explicit edits."""
    for field, value in (
        (("modules", "readout", "ro_length"), readout_length_us),
        (("modules", "readout", "trig_offset"), trigger_offset_us),
    ):
        if value is not None:
            tab.select(("modules", "readout"), inline=True)
            tab.set(field, value)
        else:
            retain_scalar(tab, field)


class LookbackInputs(FormModel):
    """Inputs for lookback; all units and null policies are declared per field."""

    target: str | None = Field(
        None,
        description="Qubit alias for this Reset; null uses the GUI aliases without changing focus.",
    )
    reuse_tab_id: str | None = Field(
        None,
        description="Idle matching tab to reset; null opens a new tab. May be combined with target.",
    )
    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the GUI selection. Frequency uses a valid captured library value, then resonator calibration.",
    )
    use_reset: str | Literal["off"] = Field(
        "off",
        description="Reset library key, or off to disable reset. Omission disables reset; null is rejected.",
    )
    init_pulse_ref: str | Literal["off"] = Field(
        "off",
        description="Initialization library key, or off to disable preparation. Omission disables it; null is rejected.",
    )
    frequency_mhz: float | None = Field(
        None,
        description="Readout frequency in MHz; explicit wins, then an explicitly selected readout library, then $resonator.freq.",
    )
    readout_length_us: float | None = Field(
        None, description="ADC acquisition window in us; null keeps the GUI value."
    )
    trigger_offset_us: float | None = Field(
        None, description="ADC trigger offset in us; null keeps the GUI value."
    )
    rounds: int | None = Field(
        None, description="Averaging round count; null keeps the GUI value."
    )


class LookbackReport(FormModel):
    """Completed lookback results; cancellation returns no report."""

    trigger_offset_us: float | None = Field(
        description="Proposed ADC trigger offset in us; null means not estimable."
    )
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )


class LookbackSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    predict_offset: float | None = Field(
        description="Owner predict_offset in us; null retains an unavailable estimate."
    )


@recipe(
    "lookback",
    description="Capture readout arrival timing in lookback; analyze, ask before writeback, and save.",
)
async def lookback(session: RecipeSession, inputs: LookbackInputs) -> LookbackReport:
    """Capture the readout pulse's arrival and ADC time envelope.

    Apply optional reset/init pulses, then read out at a fixed frequency.
    Primary estimates the trigger offset from the rising signal envelope.
    Ask before applying the timing proposal, then save raw trace and analysis.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Owner, acquisition, save and analysis errors
    propagate. Save raw data and completed artifacts once in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "lookback",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    select_or_disable(tab, ("modules", "init_pulse"), inputs.init_pulse_ref)
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
        session,
        inputs.frequency_mhz,
        calibration=FrequencyCalibration(
            path="$resonator.freq",
            required="frequency_mhz",
            prefer_library=inputs.readout_ref is not None,
        ),
    )
    if frequency.missing:
        raise RecipeNeedsParameters(frequency.missing)
    if frequency.edit is not None:
        tab.select(("modules", "readout"), inline=True)
        apply_frequency(tab, frequency.edit)
    _set_readout_timing(tab, inputs.readout_length_us, inputs.trigger_offset_us)
    tab.set(("rounds",), inputs.rounds)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Analyze the acquisition with the GUI parameters.
        analysis = await tab.analyze("primary", result=LookbackSummary)

        # Write only the proposal items selected in the answer.
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
