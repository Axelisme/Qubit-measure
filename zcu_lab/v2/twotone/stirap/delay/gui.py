"""Measure-GUI attachment for the independent STIRAP delay experiment."""

from pathlib import Path

from zcu_tools.experiment import RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import StirapDelayCfg, StirapDelayExp, StirapDelayResult, StirapDelaySummary
from .form import make_form


def _run(
    cfg: StirapDelayCfg, *, context: RunContext
) -> RunRecord[StirapDelayCfg, StirapDelayResult]:
    return RunRecord(cfg=cfg, result=StirapDelayExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[StirapDelayCfg, StirapDelayResult], None],
    *,
    plots: Plots,
) -> StirapDelaySummary:
    return StirapDelayExp().analyze(req.run_result, plots=plots)


DEFINITION = Adapter(
    spec=StirapDelayExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
