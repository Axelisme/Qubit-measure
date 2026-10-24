"""Mux GE primary/post hooks using paired raw shots and captured destinations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    PostAnalysis,
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2.singleshot.ge.core import GEAnalyzeOptions, GEPostAnalyzeOptions
from zcu_lab.v2.singleshot.ge.gui import (
    GEAnalyzeParams,
    GEPostSummary,
    ge_calibration_review_note,
    ge_post_review_note,
)

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
    """Propose complete successful tones' primary IQ calibrations, never Apply.

    Resolve destinations only from this Run/Load capture; no capture means
    no proposals. Validate the binding against any source cfg. Preserve GE's
    calibration validity gate. Failed/partial rows contribute no items.
    GE model-review gates control initial selection with a note; manual draft
    changes remain possible. Low joint r2 is not calibration evidence.
    """
    snapshot = req.run_result.cfg
    names = None if snapshot is None else tuple(t.name for t in snapshot.tones)
    qubits = captured_qubits(req.ctx.parameters, names)
    if qubits is None:
        return []
    items: list[WritebackItem] = []
    for row, outcome in enumerate(req.analyze_result.outcomes):
        if (
            isinstance(outcome, MuxFitFailure)
            or not np.isfinite(req.run_result.result.signals[row]).all()
        ):
            continue
        primary = outcome.primary
        primary.validate_calibration()
        note = ge_calibration_review_note(primary)
        qubit = qubits[outcome.index]
        row_items: list[WritebackItem] = []
        for field, value, description in (
            ("fidelity", primary.fidelity, "Prepared-label assignment fidelity"),
            ("sigma", primary.ge_s, "Single-shot IQ cluster width (ADC units)"),
            ("g_center", primary.g_center, "Single-shot ground IQ center"),
            ("e_center", primary.e_center, "Single-shot excited IQ center"),
        ):
            row_items.append(
                ParameterWriteback(
                    target_name=f"{qubit}.ext.readout_cal.{field}",
                    description=description,
                    proposed_value=value,
                    selected=note is None,
                    note=note,
                )
            )
        for item in row_items:
            item.element = ElementIdentity(index=outcome.index, name=qubit)
        items.extend(row_items)
    return items


def _get_post_writeback_items(
    req: PostWritebackRequest[
        MuxGERunResult, MuxGEAnalyzeResult, MuxGEPostAnalyzeResult
    ],
) -> Sequence[WritebackItem]:
    """Propose only same-source successful GE radius/confusion diagnostics.

    Missing capture returns no items. Invalid bindings raise; failed/partial
    rows contribute nothing. Targets retain GE's 3x3
    diagnostic policy. Preselection requires primary GE review gates and a
    finite post condition number <=10. Failed gates keep candidates with notes
    for manual selection. No current Form/library lookup or Apply occurs here.
    """
    result = req.post_analyze_result
    snapshot = req.run_result.cfg
    names = None if snapshot is None else tuple(t.name for t in snapshot.tones)
    qubits = captured_qubits(req.ctx.parameters, names)
    if qubits is None:
        return []
    items: list[WritebackItem] = []
    for row, (outcome, primary) in enumerate(
        zip(result.outcomes, req.analyze_result.outcomes, strict=True)
    ):
        if (
            isinstance(outcome, MuxFitFailure)
            or isinstance(primary, MuxFitFailure)
            or not np.isfinite(req.run_result.result.signals[row]).all()
        ):
            continue
        qubit = qubits[outcome.index]
        note = ge_post_review_note(
            primary.primary, outcome.post.confusion.condition_number
        )
        row_items: list[WritebackItem] = []
        row_items.extend(
            [
                ParameterWriteback(
                    target_name=f"{qubit}.ext.readout_cal.radius",
                    description="Single-shot classification radius",
                    proposed_value=outcome.post.confusion.radius,
                    selected=note is None,
                    note=note,
                ),
                ParameterWriteback(
                    target_name=f"{qubit}.ext.readout_cal.confusion_matrix",
                    description="Single-shot 3x3 confusion matrix",
                    proposed_value=outcome.post.confusion.matrix.tolist(),
                    selected=note is None,
                    note=note,
                ),
            ]
        )
        for item in row_items:
            item.element = ElementIdentity(index=outcome.index, name=qubit)
        items.extend(row_items)
    return items


def run(cfg: MuxGECfg, *, context: RunContext) -> RunRecord[MuxGECfg, MuxGEResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=MuxGEExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=MuxGEExp.AXES_SPEC,
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
