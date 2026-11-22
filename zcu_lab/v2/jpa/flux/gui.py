"""jpa/flux GUI adapter — JPA flux-device calibration sweep.

Owns the jpa/flux cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.flux.core import FluxAnalysis, FluxCfg, FluxExp, FluxResult
from zcu_lab.v2.jpa.flux.form import make_form

JpaFluxRunResult: TypeAlias = RunRecord[FluxCfg, FluxResult]


def _analyze(
    req: AnalyzeRequest[JpaFluxRunResult, None], *, plots: Plots
) -> FluxAnalysis:
    return FluxExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[JpaFluxRunResult, FluxAnalysis],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback(
            target_name="$jpa.flux",
            description="Best JPA flux device value",
            proposed_value=req.analyze_result.best_flux,
        )
    ]


def _summary_lines(source: JpaFluxRunResult) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


def run(cfg: FluxCfg, *, context: RunContext) -> RunRecord[FluxCfg, FluxResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=FluxExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FluxExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, writeback=_get_writeback_items, summary_lines=_summary_lines
    ),
)
