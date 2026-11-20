"""Measure GUI wiring for STIRAP detuning and saved-readout analysis."""

from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import (
    StirapDetuningCfg,
    StirapDetuningExp,
    StirapDetuningResult,
    StirapDetuningSummary,
)
from .form import make_form


def _run(
    cfg: StirapDetuningCfg, *, context: RunContext
) -> RunRecord[StirapDetuningCfg, StirapDetuningResult]:
    return RunRecord(cfg=cfg, result=StirapDetuningExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[StirapDetuningCfg, StirapDetuningResult], None],
    *,
    plots: Plots,
) -> StirapDetuningSummary:
    return StirapDetuningExp().analyze(req.run_result, None, plots=plots)


DEFINITION = Adapter(
    spec=StirapDetuningExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
