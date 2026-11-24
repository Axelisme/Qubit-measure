"""Measure attachment for paired RB with a joint four-class return fit."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.paired_rb_analysis import RBOptions

from .core import (
    RBHeraldedAnalysis,
    RBHeraldedCfg,
    RBHeraldedExp,
    RBHeraldedResult,
)
from .form import make_form

RBHeraldedRun: TypeAlias = RunRecord[RBHeraldedCfg, RBHeraldedResult]


def run(cfg: RBHeraldedCfg, *, context: RunContext) -> RBHeraldedRun:
    """Acquire heralded counts and retain the accepted cfg."""
    return RunRecord(cfg=cfg, result=RBHeraldedExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RBHeraldedRun, None], *, plots: Plots
) -> RBHeraldedAnalysis:
    return RBHeraldedExp().analyze(req.run_result, RBOptions(), plots=plots)


def _summary_lines(source: RBHeraldedRun) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps × Rounds: {cfg.reps} × {cfg.rounds}",)


DEFINITION = Adapter(
    spec=RBHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
