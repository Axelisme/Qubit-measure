"""Measure attachment for paired RB with joint gg/ee return-survival fits."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .analysis import RBOptions
from .core_heralded import (
    RBHeraldedAnalysis,
    RBHeraldedCfg,
    RBHeraldedExp,
    RBHeraldedResult,
)
from .form import rb_form

RBHeraldedRun: TypeAlias = RunRecord[RBHeraldedCfg, RBHeraldedResult]


def make_form(ctx: FormContext) -> Form[RBHeraldedCfg]:
    """Capture roles, X90/CZ/readout and named corrections; reset starts None."""
    return rb_form(ctx, RBHeraldedCfg)


def run(cfg: RBHeraldedCfg, *, context: RunContext) -> RBHeraldedRun:
    """Acquire heralded counts and retain the accepted cfg."""
    return RunRecord(cfg=cfg, result=RBHeraldedExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[RBHeraldedRun, None], *, plots: Plots
) -> RBHeraldedAnalysis:
    return RBHeraldedExp().analyze(req.run_result, RBOptions(), plots=plots)


DEFINITION = Adapter(
    spec=RBHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide_heralded.md"),
    analysis=Analysis(fn=_analyze),
)
