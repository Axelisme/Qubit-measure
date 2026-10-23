"""Readout arrival-time trace and proposed ADC trigger timing."""

from typing import Literal

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)
from zcu_tools.mcp.measure.recipe_contracts import RecipeNeedsParameters

from zcu_lab.v2.lookback.gui import LookbackSummary

from .helpers import (
    FrequencyCalibration,
    choose_frequency,
)


class LookbackInputs(RecipeInputs):
    """Inputs for lookback; all units and null policies are declared per field."""

    readout_ref: str | None = Field(
        None,
        description="Readout library key; null keeps the GUI selection. Frequency uses a valid captured library value, then resonator calibration.",
    )
    use_reset: Literal["off"] = Field(
        "off",
        description="Only off is supported: the lookback Form hides reset and does not require a reset slot.",
    )
    init_pulse_ref: Literal["off"] = Field(
        "off",
        description="Only off is supported: the lookback Form hides the initialization pulse.",
    )
    frequency_mhz: float | None = Field(
        None,
        description="Readout frequency in MHz; explicit wins, then an explicitly selected readout library, then $resonator.freq. Required when neither the selected library nor a resonator frequency source supplies a frequency.",
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

    ratio: float | None = Field(
        None,
        description="Threshold ratio; null keeps the GUI analysis value. Analysis only.",
    )
    smooth: float | None = Field(
        None,
        description="Smooth sigma; null keeps the GUI analysis value. Analysis only.",
    )
    plot_fit: bool | None = Field(
        None, description="Plot fit; null keeps the GUI analysis value. Analysis only."
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
    Analysis controls: ratio, smooth, plot_fit. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_recipe_tab("lookback", inputs, target_alias="qubit")
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select_sources(inputs)

    edits: list[FieldEdit] = []
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
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
        edits.append(frequency.edit)
    for field, value in (
        (("modules", "readout", "ro_length"), inputs.readout_length_us),
        (("modules", "readout", "trig_offset"), inputs.trigger_offset_us),
    ):
        if value is not None:
            tab.select(("modules", "readout"), inline=True)
            edits.append(FieldEdit(path=field, value=value))
    tab.apply_inputs(inputs, {("rounds",): inputs.rounds}, edits=edits)

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary",
            result=LookbackSummary,
            updates={
                key: value
                for key, value in {
                    "ratio": inputs.ratio,
                    "smooth": inputs.smooth,
                    "plot_fit": inputs.plot_fit,
                }.items()
                if value is not None
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
