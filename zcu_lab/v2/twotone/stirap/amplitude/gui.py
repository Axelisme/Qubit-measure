"""Connect the amplitude core, form and saved-readout analysis."""

from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import (
    StirapAmplitudeCfg,
    StirapAmplitudeExp,
    StirapAmplitudeResult,
    StirapAmplitudeSummary,
)
from .form import make_form


def _run(
    cfg: StirapAmplitudeCfg, *, context: RunContext
) -> RunRecord[StirapAmplitudeCfg, StirapAmplitudeResult]:
    return RunRecord(cfg=cfg, result=StirapAmplitudeExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RunRecord[StirapAmplitudeCfg, StirapAmplitudeResult], None],
    *,
    plots: Plots,
) -> StirapAmplitudeSummary:
    return StirapAmplitudeExp().analyze(req.run_result, req.analyze_params, plots=plots)


DEFINITION = Adapter(
    spec=StirapAmplitudeExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
