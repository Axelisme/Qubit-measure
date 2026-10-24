"""Measure and display flux-transient phase tomography in the measure GUI."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from .core import PhaseAnalysis, PhaseCfg, PhaseExp, PhaseResult
from .form import make_form

PhaseRunResult: TypeAlias = RunRecord[PhaseCfg, PhaseResult]


def _analyze(
    req: AnalyzeRequest[PhaseRunResult, None], *, plots: Plots
) -> PhaseAnalysis:
    """Plot tomography and unwrapped phase using only the source run timing."""
    return PhaseExp().analyze(req.run_result, None, plots=plots)


def run(cfg: PhaseCfg, *, context: RunContext) -> RunRecord[PhaseCfg, PhaseResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=PhaseExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=PhaseExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
