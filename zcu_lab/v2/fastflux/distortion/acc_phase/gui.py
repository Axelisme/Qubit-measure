"""Measure GUI attachment for accumulated-phase flux distortion."""

from __future__ import annotations

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

from .core import AccPhaseCfg, AccPhaseExp, AccPhaseResult
from .form import make_form

AccPhaseRunResult: TypeAlias = RunRecord[AccPhaseCfg, AccPhaseResult]


def _analyze(
    req: AnalyzeRequest[AccPhaseRunResult, None],
    *,
    plots: Plots,
) -> None:
    """Plot phase projections and derived detuning from the source record."""
    AccPhaseExp().analyze(req.run_result, None, plots=plots)
    return


def run(
    cfg: AccPhaseCfg, *, context: RunContext
) -> RunRecord[AccPhaseCfg, AccPhaseResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=AccPhaseExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=AccPhaseExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
