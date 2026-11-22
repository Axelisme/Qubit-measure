"""jpa/auto_optimize GUI adapter — joint JPA flux/frequency/power optimization.

Owns the jpa/auto_optimize cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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

from zcu_lab.v2.jpa.auto_optimize.core import (
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeAnalysis,
    JPAOptimizeResult,
)
from zcu_lab.v2.jpa.auto_optimize.form import make_form

JpaAutoRunResult = RunRecord[JPAOptCfg, JPAOptimizeResult]


def _analyze(
    req: AnalyzeRequest[JpaAutoRunResult, None], *, plots: Plots
) -> JPAOptimizeAnalysis:
    return AutoOptimizeExp().analyze(req.run_result, None, plots=plots)


def _get_writeback_items(
    req: WritebackRequest[JpaAutoRunResult, JPAOptimizeAnalysis],
) -> Sequence[WritebackItem]:
    result = req.analyze_result
    return [
        ParameterWriteback(
            target_name="$jpa.flux",
            description="Best JPA flux device value",
            proposed_value=result.best_flux,
        ),
        ParameterWriteback(
            target_name="$jpa.pump_freq",
            description="Best JPA pump frequency (MHz)",
            proposed_value=result.best_freq,
        ),
        ParameterWriteback(
            target_name="$jpa.pump_power",
            description="Best JPA pump power (dBm)",
            proposed_value=result.best_power,
        ),
    ]


def _summary_lines(source: JpaAutoRunResult) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


def run(
    cfg: JPAOptCfg, *, context: RunContext
) -> RunRecord[JPAOptCfg, JPAOptimizeResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=AutoOptimizeExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=AutoOptimizeExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, writeback=_get_writeback_items, summary_lines=_summary_lines
    ),
)
