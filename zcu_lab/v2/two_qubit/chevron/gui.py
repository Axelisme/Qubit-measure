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

from .._rabi_analysis import RabiAnalysis
from .._rabi_scan import RabiScanResult
from .core import ChevronCfg, ChevronExp
from .form import make_form

ChevronRun: TypeAlias = RunRecord[ChevronCfg, RabiScanResult]


def _analyze(req: AnalyzeRequest[ChevronRun, None], *, plots: Plots) -> RabiAnalysis:
    """Analyze only saved source cfg/raw, never the current Form or library."""
    return ChevronExp().analyze(req.run_result, None, plots=plots)


def run(
    cfg: ChevronCfg, *, context: RunContext
) -> RunRecord[ChevronCfg, RabiScanResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=ChevronExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=ChevronExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
