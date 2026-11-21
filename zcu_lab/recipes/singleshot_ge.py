"""Probe-off/on IQ shots with Primary and Post discrimination analysis."""

from typing import Literal

from pydantic import Field
from zcu_tools.analysis.fitting import FitQuality
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

from zcu_lab.v2.singleshot.ge.gui import GEPostSummary, GESummary


class SingleshotGEInputs(RecipeInputs):
    """Inputs for singleshot ge; all units and null policies are declared per
    field."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Sets Run › Readout to a readout library key. Omit to retain; null is passed to the form.",
    )
    pi_ref: str | None | Keep = Field(
        KEEP,
        description="Sets Run › π probe pulse to a calibrated g↔e pulse key. Omit to retain.",
    )
    use_reset: str | None | Keep = Field(
        KEEP,
        description="Sets Run › Reset to a reset module option. Omit to keep the current source; null skips reset.",
    )
    init_pulse_ref: str | None | Keep = Field(
        KEEP,
        description="Sets Run › Initial preparation pulse to a pulse option. Omit to retain; null skips it.",
    )
    shots: int | None | Keep = Field(
        KEEP,
        description="Sets Run › Shots per preparation (count) for each probe-off/on branch. Omit to retain.",
    )

    initial_state: Literal["ground", "excited"] | None | Keep = Field(
        KEEP,
        description="Sets Primary › State before π probe to ground or excited. Omit to retain.",
    )
    backend: Literal["pca", "center"] | None | Keep = Field(
        KEEP,
        description="Sets Primary › IQ rotation method: pooled axis (pca) or line between centers (center).",
    )
    logscale: bool | None | Keep = Field(
        KEEP,
        description="Sets Primary › Log scale for projected shot counts. Only the histogram display changes.",
    )
    align_t1: bool | None | Keep = Field(
        KEEP,
        description="Sets Primary › Share decay ratio across histogram fits. Omit to retain.",
    )
    length_ratio: float | None | Keep = Field(
        KEEP,
        description="Sets Primary › Readout length / T1 (a.u.). Omit to retain; null lets analysis estimate the ratio.",
    )


class SingleshotGEReport(RecipeReport):
    """Single-shot GE results and failure reason; cancellation returns no report."""

    ground_fit_quality: FitQuality | None = Field(
        default=None,
        description="Owner ground histogram fit diagnostics; null when unavailable.",
    )
    excited_fit_quality: FitQuality | None = Field(
        default=None,
        description="Owner excited histogram fit diagnostics; null when unavailable.",
    )

    fidelity: float | None = Field(
        description="Prepared-label assignment fidelity (fraction, 0–1); null means not estimable."
    )
    theta: float | None = Field(
        description="IQ rotation angle (rad); null means not estimable."
    )
    threshold: float | None = Field(
        description="Rotated I threshold (a.u.); null means not estimable."
    )
    ge_radius: float | None = Field(
        description="Classification radius (a.u.) from Post; null means not estimable."
    )
    confusion: list[list[float | None]] = Field(
        description="Corrected confusion from Post, rows model states, columns assigned outcomes, each in g/e/other order; nonfinite probabilities are null."
    )
    primary_writeback_applied: bool = Field(
        default=False, description="Primary receipt confirmed one or more writes."
    )
    post_writeback_applied: bool = Field(
        default=False, description="Post receipt confirmed one or more writes."
    )


@recipe(
    "singleshot_ge",
    description="Acquire π probe-off/on IQ clouds, fit g/e discrimination in Primary and derive circle classification in Post; ask before writing calibration and save both stages.",
)
async def singleshot_ge(
    session: RecipeSession, inputs: SingleshotGEInputs
) -> SingleshotGEReport:
    """Measure probe-off/on IQ clouds for ground/excited discrimination.

    Apply optional reset/init pulses, then acquire shots without and with the
    calibrated pi probe before readout. Primary estimates IQ rotation,
    threshold, cloud width and discrimination fidelity; Post adds classification
    radius and confusion diagnostics. Ask about each stage's calibration
    after its analysis, then save raw shots and both analyses.

    No n → expts input: shots sets the probe-off/on cloud size.
    Analysis controls: initial_state, backend, logscale, align_t1, length_ratio. Omission retains each analysis value; null is passed to its owner.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "singleshot/ge",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("reset", inputs.use_reset)
    tab.select("init_pulse", inputs.init_pulse_ref)
    tab.select("readout", inputs.readout_ref)
    tab.select("probe_pulse", inputs.pi_ref)
    tab.set("shots", inputs.shots)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        primary = await tab.analyze(
            "primary",
            result=GESummary,
            updates={
                "initial_state": inputs.initial_state,
                "backend": inputs.backend,
                "logscale": inputs.logscale,
                "align_t1": inputs.align_t1,
                "length_ratio": inputs.length_ratio,
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
        quality = primary.result.fit_quality
        primary_writeback_applied = False
        primary_answer = await tab.ask_writeback(stage="primary")
        if primary_answer.accepted:
            receipt = tab.accept_writeback(primary_answer.items)
            receipt.raise_for_error()
            primary_writeback_applied = bool(receipt.items)

        post = await tab.analyze("post", result=GEPostSummary)
        if post.result is None:
            return SingleshotGEReport(
                fit_quality=None if quality is None else quality["joint"],
                ground_fit_quality=None if quality is None else quality["ground"],
                excited_fit_quality=None if quality is None else quality["excited"],
                fidelity=report_number(primary.result.fidelity),
                theta=report_number(primary.result.theta),
                threshold=report_number(primary.result.threshold),
                ge_radius=None,
                confusion=[],
                writeback_applied=primary_writeback_applied,
                primary_writeback_applied=primary_writeback_applied,
                analysis_failure=post.reason,
            )

        post_writeback_applied = False
        post_answer = await tab.ask_writeback(stage="post")
        if post_answer.accepted:
            receipt = tab.accept_writeback(post_answer.items)
            receipt.raise_for_error()
            post_writeback_applied = bool(receipt.items)

        return SingleshotGEReport(
            fit_quality=None if quality is None else quality["joint"],
            ground_fit_quality=None if quality is None else quality["ground"],
            excited_fit_quality=None if quality is None else quality["excited"],
            fidelity=report_number(primary.result.fidelity),
            theta=report_number(primary.result.theta),
            threshold=report_number(primary.result.threshold),
            ge_radius=report_number(post.result.confusion.radius),
            confusion=post.result.confusion.matrix,
            writeback_applied=primary_writeback_applied or post_writeback_applied,
            primary_writeback_applied=primary_writeback_applied,
            post_writeback_applied=post_writeback_applied,
        )
    finally:
        tab.save()


DEFINITION = singleshot_ge
