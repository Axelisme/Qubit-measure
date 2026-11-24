"""Measure attachment for heralded idle ZZ; acquisition never writes calibration."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from ._analysis import IdleZZHeraldedAnalysis
from .core import IdleZZHeraldedCfg, IdleZZHeraldedExp, IdleZZHeraldedResult
from .form import make_form

IdleZZHeraldedRun: TypeAlias = RunRecord[IdleZZHeraldedCfg, IdleZZHeraldedResult]


def run(
    cfg: IdleZZHeraldedCfg, *, context: RunContext
) -> RunRecord[IdleZZHeraldedCfg, IdleZZHeraldedResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=IdleZZHeraldedExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[IdleZZHeraldedRun, None], *, plots: Plots
) -> IdleZZHeraldedAnalysis:
    return IdleZZHeraldedExp().analyze(req.run_result, None, plots=plots)


def _summary_lines(source: IdleZZHeraldedRun) -> tuple[str, ...]:
    cfg = source.cfg
    return (
        () if cfg is None else (f"Reps×Rounds {cfg.reps}×{cfg.rounds} per preparation",)
    )


DEFINITION = Adapter(
    spec=IdleZZHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
