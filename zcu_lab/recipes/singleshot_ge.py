"""Probe-off/on IQ shots with Primary and Post discrimination analysis."""

from typing import Literal

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from .helpers import (
    FrequencyCalibration,
    apply_frequency,
    choose_frequency,
    report_number,
    require_pi,
    select_or_disable,
)


class SingleshotGEInputs(FormModel):
    """Inputs for singleshot ge; all units and null policies are declared per
    field."""

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
    pi_ref: str | None = Field(
        None,
        description="Calibrated pi-pulse library key; null keeps the GUI selection, which must be a valid reference.",
    )
    use_reset: str | Literal["off"] = Field(
        "off",
        description="Reset library key, or off to disable reset. Omission disables reset; null is rejected.",
    )
    init_pulse_ref: str | Literal["off"] = Field(
        "off",
        description="Initialization library key, or off to disable preparation. Omission disables it; null is rejected.",
    )
    shots: int | None = Field(
        None, description="Probe-off/on shot count; null keeps the GUI value."
    )


class SingleshotGEReport(FormModel):
    """Single-shot GE results and failure reason; cancellation returns no report."""

    fidelity: float | None = Field(
        description="Ground/excited discrimination fidelity; null means not estimable."
    )
    theta: float | None = Field(
        description="IQ rotation angle in radians; null means not estimable."
    )
    threshold: float | None = Field(
        description="Rotated-IQ discrimination threshold; null means not estimable."
    )
    ge_radius: float | None = Field(
        description="Post-analysis classification radius in IQ units; null means not estimable."
    )
    confusion: list[list[float]] = Field(
        description="Post-analysis confusion matrix in state order."
    )
    writeback_applied: bool = Field(
        description="Whether the selected calibration proposal was explicitly accepted and applied."
    )

    analysis_failure: str | None = Field(
        default=None,
        description="Analysis failure reason; null when analysis completed.",
    )


class GESummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    fidelity: float | None = Field(
        description="Owner fidelity in dimensionless; null retains an unavailable estimate."
    )

    theta: float | None = Field(
        description="Owner theta in rad; null retains an unavailable estimate."
    )

    threshold: float | None = Field(
        description="Owner threshold in ADC units; null retains an unavailable estimate."
    )


class GEPostSummary(BaseModel):
    """Author view of owner summary fields; no missing values are invented."""

    ge_radius: float | None = Field(
        description="Owner ge_radius in ADC units; null retains an unavailable estimate."
    )

    confusion: list[list[float]] = Field(description="Owner confusion values.")


@recipe(
    "singleshot_ge",
    description="Run GE single-shot calibration with Primary and Post analysis.",
)
async def singleshot_ge(
    session: RecipeSession, inputs: SingleshotGEInputs
) -> SingleshotGEReport:
    """Measure probe-off/on IQ clouds for ground/excited discrimination.

    Apply optional reset/init pulses, then acquire shots without and with the
    calibrated pi probe before readout. Primary estimates IQ rotation,
    threshold, cloud width and discrimination fidelity; Post adds classification
    radius and confusion diagnostics. Ask before applying their calibration
    proposals in Primary-to-Post order, then save raw shots and both analyses.

    Inputs declare native units and GUI-retention policies. The session supplies
    fixed Form sources. Return the typed report after completion.
    The runtime owns cancellation. Expected analysis failures return a reason in
    the report; other errors propagate. Save raw data and completed artifacts once
    in the Run finalizer.
    Missing calibration raises RecipeNeedsParameters before Run.
    """
    # Select the captured sources before editing measurement fields.
    tab = session.open_tab(
        "singleshot/ge",
        aliases=None if inputs.target is None else {"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    select_or_disable(tab, ("modules", "init_pulse"), inputs.init_pulse_ref)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    missing: list[MissingParameter] = []
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
        session,
        calibration=FrequencyCalibration(
            path="$resonator.freq", required="readout_ref"
        ),
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.select(("modules", "readout"), inline=True)
        apply_frequency(tab, frequency.edit)
    if (
        parameter := require_pi(
            tab, ("modules", "probe_pulse"), inputs.pi_ref, required="pi_ref"
        )
    ) is not None:
        missing.append(parameter)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    tab.set(("shots",), inputs.shots)

    # Acquire before analysis; the finally block saves raw and artifacts.
    await tab.run()
    try:
        # Fit IQ clouds before deriving Post diagnostics.
        primary = await tab.analyze("primary", result=GESummary)
        if primary.result is None:
            return SingleshotGEReport(
                fidelity=None,
                theta=None,
                threshold=None,
                ge_radius=None,
                confusion=[],
                writeback_applied=False,
                analysis_failure=primary.reason,
            )
        analysis = await tab.analyze("post", result=GEPostSummary)
        if analysis.result is None:
            return SingleshotGEReport(
                fidelity=report_number(primary.result.fidelity),
                theta=report_number(primary.result.theta),
                threshold=report_number(primary.result.threshold),
                ge_radius=None,
                confusion=[],
                writeback_applied=False,
                analysis_failure=analysis.reason,
            )

        # Write only the proposal items selected in the answer.
        decision = await tab.ask_writeback()
        writeback_applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            writeback_applied = bool(receipt.items)
        return SingleshotGEReport(
            fidelity=report_number(primary.result.fidelity),
            theta=report_number(primary.result.theta),
            threshold=report_number(primary.result.threshold),
            ge_radius=report_number(analysis.result.ge_radius),
            confusion=analysis.result.confusion,
            writeback_applied=writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = singleshot_ge
