"""GUI wiring for Raman count scans and saved-readout population analysis."""

from pathlib import Path

from zcu_tools.analysis.singleshot import Centroid3IQAnalysis
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import RamanGFCfg, RamanGFExp, RamanGFResult
from .form import make_form


def _run(
    cfg: RamanGFCfg, *, context: RunContext
) -> RunRecord[RamanGFCfg, RamanGFResult]:
    return RunRecord(cfg=cfg, result=RamanGFExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[RamanGFCfg, RamanGFResult], None],
    *,
    plots: Plots,
) -> Centroid3IQAnalysis:
    return RamanGFExp().analyze(req.run_result, None, plots=plots)


DEFINITION = Adapter(
    spec=RamanGFExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
