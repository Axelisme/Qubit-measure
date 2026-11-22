"""Mux GE primary/post hooks using paired raw shots and captured destinations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.cfg_model import CircleClassifier
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParamMeta,
    PostAnalysis,
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.writeback import calibrated_module_writeback
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2.support.measure.mux_save_names import mux_save_stem
from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.singleshot.center_quality import (
    GE_CENTER_ERROR_PATHS,
    ge_center_relative_error_reasons,
)

from .core import (
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
    MuxGEAnalysis,
    MuxGEAnalyzeOptions,
    MuxGECfg,
    MuxGEExp,
    MuxGEPostAnalysis,
    MuxGEPostOptions,
    MuxGEResult,
)
from .form import make_form

MuxGERunResult: TypeAlias = RunRecord[MuxGECfg, MuxGEResult]


@dataclass
class GEAnalyzeParams:
    """GUI controls for fitting paired probe-off/on IQ clouds.

    initial_state identifies probe-off as predominantly ground or excited.
    backend selects PCA or center rotation for the primary rotation/threshold fit.
    logscale uses logarithmic histogram display; align_t1 enables T1 alignment.
    length_ratio is readout length / T1, or None to estimate it from the data.
    """

    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"
    # ``backend`` selects the primary rotation/threshold fit. Post-analysis uses
    # the resulting centres and does not choose or run another fit backend.
    backend: Annotated[Literal["pca", "center"], ParamMeta(label="Backend")] = "pca"
    logscale: Annotated[bool, ParamMeta(label="Log Scale")] = False
    align_t1: Annotated[bool, ParamMeta(label="Align T1")] = True
    length_ratio: Annotated[
        float | None, ParamMeta(label="Length ratio (a.u.)", decimals=4)
    ] = None


class GEConfusionSummary(BaseModel):
    """Consumed post-calibration JSON fields; nonfinite estimates arrive as null."""

    radius: float | None = Field(
        description="Classification radius in ADC units; null means unavailable."
    )
    matrix: list[list[float | None]] = Field(
        description="Prepared-to-measured confusion fractions; nonfinite entries are null."
    )
    condition_number: float | None = Field(
        description="Dimensionless matrix condition number; null means unavailable."
    )


class GEPostSummary(BaseModel):
    """Decode the consumed fields of the core post-analysis JSON object."""

    confusion: GEConfusionSummary = Field(
        description="Radius, confusion matrix and condition diagnostic from the core."
    )


def ge_calibration_review_note(primary: GEAnalysis) -> str | None:
    """Explain manual selection for mux GE model-derived calibrations.

    The primary analysis's joint histogram fit must have r2 >=0.9, normalized
    residual RMS <=0.1, relative width (s) error <=0.5, and relative center
    errors (sg and se) each <=0.5. The note names failed thresholds and missing
    or nonfinite metrics. Invalid issues for required metrics also prevent
    preselection; unrelated fixed-zero parameter issues do not. None means
    these checks passed, not that pure-state readout fidelity was established.
    """
    quality = None if primary.fit_quality is None else primary.fit_quality.get("joint")
    if quality is None:
        return "Review required: joint histogram quality unavailable"
    reasons = [
        f"joint {issue.path}: {issue.reason}"
        for issue in quality.invalid
        if issue.path in GE_CENTER_ERROR_PATHS
        or issue.path
        in {"r2", "normalized_residual_rms", "relative_parameter_errors.s"}
    ]
    if quality.r2 is None or not math.isfinite(quality.r2):
        reasons.append("joint r² unavailable")
    elif quality.r2 < 0.9:
        reasons.append(f"joint r²={quality.r2:.4g} <0.9")
    rms = quality.normalized_residual_rms
    if rms is None or not math.isfinite(rms):
        reasons.append("joint normalized RMS unavailable")
    elif rms > 0.1:
        reasons.append(f"joint normalized RMS={rms:.4g} >0.1")
    width_error = quality.relative_parameter_errors.get("s")
    if width_error is None or not math.isfinite(width_error):
        reasons.append("relative width error unavailable")
    elif width_error > 0.5:
        reasons.append(f"relative width error={width_error:.4g} >0.5")
    reasons.extend(ge_center_relative_error_reasons(quality))
    return "Review required: " + "; ".join(reasons) if reasons else None


def ge_post_review_note(primary: GEAnalysis, condition_number: float) -> str | None:
    """Guard radius/confusion by primary GE quality and finite condition <=10.

    The condition is dimensionless. Manual draft selection remains possible;
    None means the review gates passed, not that the correction is reliable.
    """
    primary_note = ge_calibration_review_note(primary)
    reasons = [] if primary_note is None else [primary_note]
    if not math.isfinite(condition_number):
        reasons.append("post condition number unavailable")
    elif condition_number > 10:
        reasons.append(f"post condition number={condition_number:.4g} >10")
    return "; ".join(reasons) if reasons else None


class MuxGEPrimarySummary(BaseModel):
    """Native state, fidelity, radian rotation and normalized-ADC GE estimates.

    init_pops holds initial state populations. Null retains nonfinite estimates.
    fit_quality retains the fitting owner's metrics and native diagnostic paths.
    """

    initial_state: str
    fidelity: float | None
    theta: float | None
    threshold: float | None
    ge_s: float | None
    init_pops: list[list[float | None]]
    fit_quality: dict[str, FitQuality] | None


class MuxGEFitSummary(MuxSummaryTone):
    """Decode the native indexed primary result."""

    primary: MuxGEPrimarySummary


class MuxGESummary(BaseModel):
    """Primary GE outcomes in native order, preserving failures and identities."""

    outcomes: list[MuxGEFitSummary | MuxSummaryFailure]
    names: list[str] | None


class MuxGEPostFitSummary(MuxSummaryTone):
    """Normalized-ADC radius and single-qubit confusion matrix, not a joint matrix."""

    post: GEPostSummary


class MuxGEPostSummary(BaseModel):
    """Post GE outcomes in their own native order with original tone identities."""

    outcomes: list[MuxGEPostFitSummary | MuxSummaryFailure]
    names: list[str] | None


@dataclass(frozen=True)
class MuxGEAnalyzeResult:
    """Primary per-index outcomes and full source names, absent without cfg."""

    outcomes: tuple[MuxGEAnalysis | MuxFitFailure, ...]
    names: tuple[str, ...] | None


@dataclass(frozen=True)
class MuxGEPostAnalyzeResult:
    """Per-index single-qubit post diagnostics and names from the adopted primary."""

    outcomes: tuple[MuxGEPostAnalysis | MuxFitFailure, ...]
    names: tuple[str, ...] | None


def _analyze(
    req: AnalyzeRequest[MuxGERunResult, GEAnalyzeParams], *, plots: Plots
) -> MuxGEAnalyzeResult:
    """Use core GE options on the explicit source, publishing fit_i figures."""
    params = req.analyze_params
    outcomes = MuxGEExp().analyze(
        req.run_result,
        MuxGEAnalyzeOptions(
            default=GEAnalyzeOptions(
                initial_state=params.initial_state,
                backend=params.backend,
                logscale=params.logscale,
                align_t1=params.align_t1,
                length_ratio=params.length_ratio,
            )
        ),
        plots=plots,
    )
    snapshot = req.run_result.cfg
    names = None if snapshot is None else tuple(t.name for t in snapshot.tones)
    return MuxGEAnalyzeResult(outcomes, names)


def _post_analyze(
    req: PostAnalyzeRequest[MuxGERunResult, MuxGEAnalyzeResult, None], *, plots: Plots
) -> MuxGEPostAnalyzeResult:
    """Apply the same-source adopted primary via GE owner, publishing post_i."""
    outcomes = MuxGEExp().post_analyze(
        req.run_result,
        req.analyze_result.outcomes,
        MuxGEPostOptions(default=GEPostAnalyzeOptions()),
        plots=plots,
    )
    return MuxGEPostAnalyzeResult(outcomes, req.analyze_result.names)


def _get_writeback_items(
    req: WritebackRequest[MuxGERunResult, MuxGEAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose one complete module with all successful tones' circle centers.

    Use the Run/Load readout slot and captured binding, never the current Form.
    Failed/partial tones keep their library classifiers. Review notes from any
    successful tone deselect the whole proposal for manual review. No Apply.
    """
    snapshot = req.run_result.cfg
    if snapshot is None or req.ctx.parameters is None:
        return []
    qubits = captured_qubits(req.ctx.parameters, tuple(t.name for t in snapshot.tones))
    if qubits is None:
        return []
    updates: dict[str, CircleClassifier] = {}
    notes: list[str] = []
    for row, outcome in enumerate(req.analyze_result.outcomes):
        if (
            isinstance(outcome, MuxFitFailure)
            or not np.isfinite(req.run_result.result.signals[row]).all()
        ):
            continue
        primary = outcome.primary
        primary.validate_calibration()
        note = ge_calibration_review_note(primary)
        if note is not None:
            notes.append(f"{qubits[outcome.index]}: {note}")
        updates[f"tones.{outcome.index}.classifier"] = CircleClassifier(
            g_center=primary.g_center, e_center=primary.e_center
        )
    if not updates:
        return []
    item = calibrated_module_writeback(
        req.ctx.parameters,
        snapshot.readout,
        target="$mux.module.readout",
        description="Mux single-shot ground/excited centers",
        field_updates=updates,
    )
    item.selected, item.note = not notes, "\n".join(notes) or None
    return [item]


def _get_post_writeback_items(
    req: PostWritebackRequest[
        MuxGERunResult, MuxGEAnalyzeResult, MuxGEPostAnalyzeResult
    ],
) -> Sequence[WritebackItem]:
    """Propose one complete module with successful same-source Post classifiers.

    Keep failed/partial tones unchanged. Every updated circle contains its
    primary centers and Post radius/3x3 prepared-to-measured confusion matrix.
    Any primary or Post review note deselects the whole proposal. No Apply.
    """
    snapshot = req.run_result.cfg
    if snapshot is None or req.ctx.parameters is None:
        return []
    qubits = captured_qubits(req.ctx.parameters, tuple(t.name for t in snapshot.tones))
    if qubits is None:
        return []
    updates: dict[str, CircleClassifier] = {}
    notes: list[str] = []
    for row, (outcome, primary) in enumerate(
        zip(req.post_analyze_result.outcomes, req.analyze_result.outcomes, strict=True)
    ):
        if (
            isinstance(outcome, MuxFitFailure)
            or isinstance(primary, MuxFitFailure)
            or not np.isfinite(req.run_result.result.signals[row]).all()
        ):
            continue
        note = ge_post_review_note(
            primary.primary, outcome.post.confusion.condition_number
        )
        if note is not None:
            notes.append(f"{qubits[outcome.index]}: {note}")
        updates[f"tones.{outcome.index}.classifier"] = CircleClassifier(
            g_center=primary.primary.g_center,
            e_center=primary.primary.e_center,
            radius=outcome.post.confusion.radius,
            confusion_matrix=outcome.post.confusion.matrix.tolist(),
        )
    if not updates:
        return []
    item = calibrated_module_writeback(
        req.ctx.parameters,
        snapshot.readout,
        target="$mux.module.readout",
        description="Mux single-shot centers, radius and confusion",
        field_updates=updates,
    )
    item.selected, item.note = not notes, "\n".join(notes) or None
    return [item]


def run(cfg: MuxGECfg, *, context: RunContext) -> RunRecord[MuxGECfg, MuxGEResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=MuxGEExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=replace(MuxGEExp.AXES_SPEC, save_stem=mux_save_stem),
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: GEAnalyzeParams(),
        writeback=_get_writeback_items,
    ),
    post_analysis=PostAnalysis(fn=_post_analyze, writeback=_get_post_writeback_items),
)
