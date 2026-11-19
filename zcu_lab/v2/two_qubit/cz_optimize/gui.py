"""Measure adapter for fixed-depth interleaved RB optimization."""

from pathlib import Path

from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import (
    CZOptimizeAnalysis,
    CZOptimizeCfg,
    CZOptimizeExp,
    CZOptimizeResult,
    CZOptimizeUnavailable,
)
from .form import make_form

type _Run = RunRecord[CZOptimizeCfg, CZOptimizeResult]


def _analyze(
    req: AnalyzeRequest[_Run, None], *, plots: Plots
) -> CZOptimizeAnalysis | CZOptimizeUnavailable:
    """Compare measured evaluations from the selected Run or Load record."""
    return CZOptimizeExp().analyze(req.run_result, plots=plots)


def run(cfg: CZOptimizeCfg, *, context: RunContext) -> _Run:
    """Run with captured calibration and retain the original cfg, not a candidate."""
    return RunRecord(cfg=cfg, result=CZOptimizeExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=CZOptimizeExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
