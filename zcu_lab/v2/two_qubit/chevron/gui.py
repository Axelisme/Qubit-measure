"""Measure adapter for chevron."""

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

from .core import ChevronCfg, ChevronExp, ChevronResult
from .form import make_form

ChevronRun: TypeAlias = RunRecord[ChevronCfg, ChevronResult]


def _analyze(req: AnalyzeRequest[ChevronRun, None], *, plots: Plots) -> None:
    """Analyze saved source cfg/counts, never the current Form or library."""
    return ChevronExp().analyze(req.run_result, None, plots=plots)


def run(
    cfg: ChevronCfg, *, context: RunContext
) -> RunRecord[ChevronCfg, ChevronResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=ChevronExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=ChevronExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
