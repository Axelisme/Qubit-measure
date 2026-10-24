"""Measure attachment for conditional Ramsey; acquisition never writes calibration."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from ._analysis import IdleZZAnalysis
from .core import IdleZZCfg, IdleZZExp, IdleZZResult
from .form import make_form

IdleZZRun: TypeAlias = RunRecord[IdleZZCfg, IdleZZResult]


def _analyze(req: AnalyzeRequest[IdleZZRun, None], *, plots: Plots) -> IdleZZAnalysis:
    """Fit saved cfg/raw only; current aliases, Form and library are ignored."""
    return IdleZZExp().analyze(req.run_result, None, plots=plots)


def run(cfg: IdleZZCfg, *, context: RunContext) -> RunRecord[IdleZZCfg, IdleZZResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=IdleZZExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=IdleZZExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
