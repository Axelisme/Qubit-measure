"""Probe-off/on IQ shots with Primary and Post discrimination analysis."""

from typing import Literal

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

from zcu_lab.v2.singleshot.ge.gui import GEPostSummary, GESummary


class SingleshotGEInputs(RecipeInputs):
    """Inputs for singleshot ge; all units and null policies are declared per
    field."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key; omission retains the Form value, null clears/disables it.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Calibrated pi-pulse library key; omission retains the Form value, null clears/disables it.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Preparation module option; omission retains the Form value, null clears/disables it.",
    )
    init_pulse_ref: str | None | Keep = Field(
        KEEP,
        description="Preparation module option; omission retains the Form value, null clears/disables it.",
    )
    shots: int | None | Keep = Field(
        KEEP,
        description="Probe-off/on shot count; omission retains the Form value, null clears/disables it.",
    )

    initial_state: Literal["ground", "excited"] | None | Keep = Field(
        KEEP,
        description="Initial State; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    backend: Literal["pca", "center"] | None | Keep = Field(
        KEEP,
        description="Backend; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    logscale: bool | None | Keep = Field(
        KEEP,
        description="Log Scale; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    align_t1: bool | None | Keep = Field(
        KEEP,
        description="Align T1; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
    )
    length_ratio: float | None | Keep = Field(
        KEEP,
        description="Length Ratio; omission retains the analysis value; null is passed to the analysis owner. Analysis only.",
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
    Analysis controls: initial_state, backend, logscale, align_t1, length_ratio. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "singleshot/ge",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("modules.reset", inputs.use_reset)
    tab.select("modules.init_pulse", inputs.init_pulse_ref)
    tab.select("modules.readout", inputs.readout_ref)
    tab.select("modules.probe_pulse", inputs.pi_ref)
    tab.set("shots", inputs.shots)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

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
                if not isinstance(value, Keep)
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
