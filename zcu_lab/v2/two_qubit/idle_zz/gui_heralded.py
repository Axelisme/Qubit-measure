"""Measure attachment for herald-conditioned idle ZZ."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core_heralded import (
    IdleZZHeraldedAnalysis,
    IdleZZHeraldedCfg,
    IdleZZHeraldedExp,
    IdleZZHeraldedResult,
)
from .form import idle_zz_form

IdleZZHeraldedRun: TypeAlias = RunRecord[IdleZZHeraldedCfg, IdleZZHeraldedResult]


def make_form(ctx: FormContext) -> Form[IdleZZHeraldedCfg]:
    """Capture roles and seed calibrated pulses/readout; reset starts None."""
    return idle_zz_form(ctx, IdleZZHeraldedCfg)


def run(cfg: IdleZZHeraldedCfg, *, context: RunContext) -> IdleZZHeraldedRun:
    """Run heralded idle ZZ and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=IdleZZHeraldedExp().run(cfg, context=context))


def _analyze(
    req: AnalyzeRequest[IdleZZHeraldedRun, None], *, plots: Plots,
) -> IdleZZHeraldedAnalysis:
    return IdleZZHeraldedExp().analyze(req.run_result, None, plots=plots)


DEFINITION = Adapter(
    spec=IdleZZHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide_heralded.md"),
    analysis=Analysis(fn=_analyze),
)
