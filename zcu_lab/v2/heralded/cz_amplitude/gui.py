"""Measure adapter for the four measured CZ return branches."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import (
    HeraldedCZAmplitudeAnalysis,
    HeraldedCZAmplitudeCfg,
    HeraldedCZAmplitudeExp,
    HeraldedCZAmplitudeResult,
)
from .form import make_form

HeraldedCZAmplitudeRun: TypeAlias = RunRecord[
    HeraldedCZAmplitudeCfg, HeraldedCZAmplitudeResult
]


def run(cfg: HeraldedCZAmplitudeCfg, *, context: RunContext) -> HeraldedCZAmplitudeRun:
    """Acquire the four nominal preparations and retain the applied cfg."""
    return RunRecord(cfg=cfg, result=HeraldedCZAmplitudeExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[HeraldedCZAmplitudeRun, None], *, plots: Plots
) -> HeraldedCZAmplitudeAnalysis:
    return HeraldedCZAmplitudeExp().analyze(req.run_result, None, plots=plots)


def _summary_lines(source: HeraldedCZAmplitudeRun) -> tuple[str, ...]:
    cfg = source.cfg
    return (
        () if cfg is None else (f"Reps×Rounds per preparation {cfg.reps}×{cfg.rounds}",)
    )


DEFINITION = Adapter(
    spec=HeraldedCZAmplitudeExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, summary_lines=_summary_lines),
)
