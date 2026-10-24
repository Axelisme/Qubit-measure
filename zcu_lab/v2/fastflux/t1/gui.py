"""Measure GUI for gain-resolved relaxation during a fast-flux pulse."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from .core import T1Analysis, T1Cfg, T1Exp, T1Result
from .form import make_form

T1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass(frozen=True)
class T1AnalyzeResult:
    """Raw gains and core decay estimates; rejected fits remain non-finite."""

    gains: list[float]
    analysis: T1Analysis


def _analyze(
    req: AnalyzeRequest[T1RunResult, None], *, plots: Plots
) -> T1AnalyzeResult:
    source = req.run_result
    return T1AnalyzeResult(
        gains=source.result.gains.tolist(),
        analysis=T1Exp().analyze(source, None, plots=plots),
    )


def run(cfg: T1Cfg, *, context: RunContext) -> RunRecord[T1Cfg, T1Result]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=T1Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=T1Exp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
