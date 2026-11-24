"""Measure adapter for classified heralded Local Z."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import (
    LocalZHeraldedAnalysis,
    LocalZHeraldedCfg,
    LocalZHeraldedExp,
    LocalZHeraldedResult,
)
from .form import make_form

LocalZHeraldedRun: TypeAlias = RunRecord[LocalZHeraldedCfg, LocalZHeraldedResult]


def run(cfg: LocalZHeraldedCfg, *, context: RunContext) -> LocalZHeraldedRun:
    """Retain the typed cfg and all acquired class counts."""
    return RunRecord(cfg=cfg, result=LocalZHeraldedExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[LocalZHeraldedRun, None], *, plots: Plots
) -> LocalZHeraldedAnalysis:
    """Use the saved cfg and counts, not the current settings."""
    return LocalZHeraldedExp().analyze(req.run_result, None, plots=plots)


def _summary_lines(source: LocalZHeraldedRun) -> tuple[str, ...]:
    cfg = source.cfg
    return () if cfg is None else (f"Reps×Rounds {cfg.reps}×{cfg.rounds}",)


DEFINITION = Adapter(
    spec=LocalZHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
