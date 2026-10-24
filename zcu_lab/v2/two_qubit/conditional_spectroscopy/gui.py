"""Measure adapter for four thermal branches of two-line spectroscopy."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .analysis import SpectroscopyAnalysis
from .core import SpectroscopyCfg, SpectroscopyExp, SpectroscopyResult
from .form import make_form

SpectroscopyRun: TypeAlias = RunRecord[SpectroscopyCfg, SpectroscopyResult]


def _analyze(
    req: AnalyzeRequest[SpectroscopyRun, None], *, plots: Plots
) -> SpectroscopyAnalysis:
    """Use only source cfg/raw; never read current parameters or propose peaks."""
    return SpectroscopyExp().analyze(req.run_result, None, plots=plots)


def run(
    cfg: SpectroscopyCfg, *, context: RunContext
) -> RunRecord[SpectroscopyCfg, SpectroscopyResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=SpectroscopyExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=SpectroscopyExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
