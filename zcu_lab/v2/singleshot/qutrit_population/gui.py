"""Measure GUI wiring for three preparation-label references."""

from pathlib import Path

from zcu_tools.analysis.singleshot import ReferenceIQAnalysis
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import QutritPopulationCfg, QutritPopulationExp, QutritPopulationResult
from .form import make_form


def _run(
    cfg: QutritPopulationCfg, *, context: RunContext
) -> RunRecord[QutritPopulationCfg, QutritPopulationResult]:
    return RunRecord(cfg=cfg, result=QutritPopulationExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[QutritPopulationCfg, QutritPopulationResult], None],
    *,
    plots: Plots,
) -> ReferenceIQAnalysis:
    return QutritPopulationExp().analyze(req.run_result, None, plots=plots)


DEFINITION = Adapter(
    spec=QutritPopulationExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
