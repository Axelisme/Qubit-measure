"""Probe-off/on IQ shots with Primary and Post discrimination analysis."""

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
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)

from zcu_lab.v2.singleshot.ge.gui import GEPostSummary, GESummary

from .helpers import (
    FrequencyCalibration,
    choose_frequency,
    require_pi,
    select_or_disable,
)


class SingleshotGEInputs(RecipeInputs):
    """Inputs for singleshot ge; all units and null policies are declared per
    field."""

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

    initial_state: Literal["ground", "excited"] | None = Field(
        None,
        description="Initial State; null keeps the GUI analysis value. Analysis only.",
    )
    backend: Literal["pca", "center"] | None = Field(
        None, description="Backend; null keeps the GUI analysis value. Analysis only."
    )
    logscale: bool | None = Field(
        None, description="Log Scale; null keeps the GUI analysis value. Analysis only."
    )
    align_t1: bool | None = Field(
        None, description="Align T1; null keeps the GUI analysis value. Analysis only."
    )
    length_ratio: float | None = Field(
        None,
        description="Length Ratio; null keeps the GUI analysis value. Analysis only.",
    )


class SingleshotGEReport(RecipeReport):
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

    No n → expts input: shots sets the probe-off/on cloud size.
    Analysis controls: initial_state, backend, logscale, align_t1, length_ratio. Null keeps each GUI value.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_recipe_tab("singleshot/ge", inputs, target_alias="qubit")
    select_or_disable(tab, ("modules", "reset"), inputs.use_reset)
    select_or_disable(tab, ("modules", "init_pulse"), inputs.init_pulse_ref)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    tab.select_sources(inputs)
    edits: list[FieldEdit] = []
    missing: list[MissingParameter] = []
    frequency = choose_frequency(
        tab,
        ("modules", "readout", "freq"),
        calibration=FrequencyCalibration(
            path="$resonator.freq", required="readout_ref"
        ),
    )
    missing.extend(frequency.missing)
    if frequency.edit is not None:
        tab.select(("modules", "readout"), inline=True)
        edits.append(frequency.edit)
    if (
        parameter := require_pi(
            tab, ("modules", "probe_pulse"), inputs.pi_ref, required="pi_ref"
        )
    ) is not None:
        missing.append(parameter)
    if missing:
        raise RecipeNeedsParameters(tuple(missing))
    tab.apply_inputs(inputs, {("shots",): inputs.shots}, edits=edits)
    await tab.run()
    try:
        primary = await tab.analyze(
            "primary",
            result=GESummary,
            updates={
                key: value
                for key, value in {
                    "initial_state": inputs.initial_state,
                    "backend": inputs.backend,
                    "logscale": inputs.logscale,
                    "align_t1": inputs.align_t1,
                    "length_ratio": inputs.length_ratio,
                }.items()
                if value is not None
            },
        )
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
