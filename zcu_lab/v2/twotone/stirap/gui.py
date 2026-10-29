"""Measure GUI wiring for two-generator STIRAP scans and saved-readout analysis."""

from pathlib import Path

from zcu_tools.analysis.singleshot import Centroid3IQAnalysis
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import StirapCfg, StirapExp, StirapResult
from .form import make_form


def _run(cfg: StirapCfg, *, context: RunContext) -> RunRecord[StirapCfg, StirapResult]:
    return RunRecord(cfg=cfg, result=StirapExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[StirapCfg, StirapResult], None],
    *,
    plots: Plots,
) -> Centroid3IQAnalysis:
    return StirapExp().analyze(req.run_result, None, plots=plots)


DEFINITION = Adapter(
    spec=StirapExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
