"""Mux GE primary/post hooks using paired raw shots and captured destinations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.cfg_model import CircleClassifier
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    PostAnalysis,
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.writeback import calibrated_module_writeback
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.singleshot.ge.core import GEAnalyzeOptions, GEPostAnalyzeOptions
from zcu_lab.v2.singleshot.ge.gui import (
    GEAnalyzeParams,
    GEPostSummary,
    ge_calibration_review_note,
    ge_post_review_note,
)
from zcu_lab.v2.support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2.support.measure.mux_save_names import mux_save_stem
from zcu_lab.v2.support.mux import MuxFitFailure

from .analysis import (
    MuxGEAnalysis,
    MuxGEAnalyzeOptions,
    MuxGEPostAnalysis,
    MuxGEPostOptions,
)
from .core import MuxGECfg, MuxGEExp, MuxGEResult
from .form import make_form

MuxGERunResult: TypeAlias = RunRecord[MuxGECfg, MuxGEResult]


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
