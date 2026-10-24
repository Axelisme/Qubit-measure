"""Mux t1 source/analysis/proposal hooks; numeric core remains GUI-free."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.core.element import ElementIdentity
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.mux_results import (
    MuxSummaryFailure,
    MuxSummaryTone,
    captured_qubits,
)
from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions

from .analysis import MuxT1Fit
from .core import MuxT1Cfg, MuxT1Exp, MuxT1Result
from .form import make_form

MuxT1RunResult: TypeAlias = RunRecord[MuxT1Cfg, MuxT1Result]


class MuxT1ValuesSummary(BaseModel):
    """Decode the native nested relaxation fit in microseconds."""

    t1: float | None = Field(
        description="Owner t1 in us; null retains an unavailable estimate."
    )

    t1_err: float | None = Field(
        description="Owner t1_err in us; null retains an unavailable estimate."
    )

    t1b: float | None = Field(
        description="Owner t1b in us; null retains an unavailable estimate."
    )

    t1b_err: float | None = Field(
        description="Owner t1b_err in us; null retains an unavailable estimate."
    )

    fit_quality: dict[str, FitQuality] | None


class MuxT1FitSummary(MuxSummaryTone):
    """Original tone index and native relaxation fit."""

    fit: MuxT1ValuesSummary


class MuxT1Summary(BaseModel):
    """Owner tone rows in analysis order, retaining successes and failures."""

    outcomes: list[MuxT1FitSummary | MuxSummaryFailure]
    names: list[str] | None


@dataclass(frozen=True)
class MuxT1AnalyzeResult:
    """Indexed core outcomes and complete source names, absent when cfg is absent."""

    outcomes: tuple[MuxT1Fit | MuxFitFailure, ...]
    names: tuple[str, ...] | None


def _MuxT1_options(result: MuxT1RunResult, ctx: SessionEnv) -> T1AnalyzeOptions:
    """Seed the analysis controls without changing the core defaults."""
    del result, ctx
    return T1AnalyzeOptions(dual_exp=False, skip=0)


def _analyze(
    req: AnalyzeRequest[MuxT1RunResult, T1AnalyzeOptions], *, plots: Plots
) -> MuxT1AnalyzeResult:
    """Analyze the explicit Run/Load source, retaining each original-index outcome."""
    outcomes = MuxT1Exp().analyze(
        req.run_result,
        req.analyze_params,
        plots=plots,
    )
    snapshot = req.run_result.cfg
    return MuxT1AnalyzeResult(
        outcomes,
        None if snapshot is None else tuple(t.name for t in snapshot.tones),
    )


def _get_writeback_items(
    req: WritebackRequest[MuxT1RunResult, MuxT1AnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose only complete, successful tones using the same Run/Load capture.

    Missing capture returns no items. Captured bindings must match any source
    cfg names/count; errors propagate. Partial/nonfinite rows and failed fits
    produce no calibration items. Pulse proposals copy source cfg only, never
    current library values. This method neither writes nor applies proposals.
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
        qubit = qubits[outcome.index]
        row_items: list[WritebackItem] = []
        fit = outcome.fit
        row_items.extend(
            [
                ParameterWriteback(
                    target_name=f"{qubit}.t1",
                    description="T1 relaxation time (us)",
                    proposed_value=fit.t1,
                    stderr=fit.t1_err,
                ),
                SampleWriteback(
                    target_name=qubit,
                    description="T1 sample at captured bias",
                    coordinate_device="$mux.flux_source",
                    proposed_value={"t1_us": fit.t1, "t1_err_us": fit.t1_err},
                ),
            ]
        )
        for item in row_items:
            item.element = ElementIdentity(index=outcome.index, name=qubit)
        items.extend(row_items)
    return items


def run(cfg: MuxT1Cfg, *, context: RunContext) -> RunRecord[MuxT1Cfg, MuxT1Result]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=MuxT1Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=MuxT1Exp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, options=_MuxT1_options, writeback=_get_writeback_items
    ),
)
