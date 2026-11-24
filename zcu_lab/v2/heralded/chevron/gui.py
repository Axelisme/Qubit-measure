"""Measure adapter for heralded chevron."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import HeraldedChevronCfg, HeraldedChevronExp, HeraldedChevronResult
from .form import make_form

HeraldedChevronRun: TypeAlias = RunRecord[HeraldedChevronCfg, HeraldedChevronResult]


def run(cfg: HeraldedChevronCfg, *, context: RunContext) -> HeraldedChevronRun:
    """Acquire herald and terminal counts and keep the applied configuration."""
    return RunRecord(cfg=cfg, result=HeraldedChevronExp().run(cfg, context=context))


def _analyze(req: AnalyzeRequest[HeraldedChevronRun, None], *, plots: Plots) -> None:
    HeraldedChevronExp().analyze(req.run_result, None, plots=plots)


def _summary_lines(source: HeraldedChevronRun) -> tuple[str, ...]:
    cfg = source.cfg
    return (
        () if cfg is None else (f"Reps×Rounds {cfg.reps}×{cfg.rounds} per preparation",)
    )


DEFINITION = Adapter(
    spec=HeraldedChevronExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
