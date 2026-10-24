"""Measure adapter for source-bound paired two-qubit RB."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .analysis import RBAnalysis, RBOptions
from .core import RBCfg, RBExp, RBResult
from .form import make_form

RBRun: TypeAlias = RunRecord[RBCfg, RBResult]


def _analyze(req: AnalyzeRequest[RBRun, None], *, plots: Plots) -> RBAnalysis:
    """Count joint00 and fit only the saved source, never the current Form."""
    return RBExp().analyze(req.run_result, RBOptions(), plots=plots)


def run(cfg: RBCfg, *, context: RunContext) -> RunRecord[RBCfg, RBResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=RBExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=RBExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
