"""Heralded measure adapter for chevron."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core_heralded import ChevronHeraldedCfg, ChevronHeraldedExp, ChevronHeraldedResult
from .form import scan_form

ChevronHeraldedRun: TypeAlias = RunRecord[ChevronHeraldedCfg, ChevronHeraldedResult]


def _analyze(req: AnalyzeRequest[ChevronHeraldedRun, None], *, plots: Plots) -> None:
    """Analyze saved source cfg/counts, never the current Form or library."""
    return ChevronHeraldedExp().analyze(req.run_result, None, plots=plots)


def run(
    cfg: ChevronHeraldedCfg, *, context: RunContext
) -> RunRecord[ChevronHeraldedCfg, ChevronHeraldedResult]:
    """Run the experiment and retain its actual configuration."""
    return RunRecord(cfg=cfg, result=ChevronHeraldedExp().run(cfg, context=context))


def make_form(ctx: FormContext) -> Form[ChevronHeraldedCfg]:
    """Capture roles and seed heralded scan inputs; optional reset starts None."""
    return scan_form(ctx, ChevronHeraldedCfg)


DEFINITION = Adapter(
    spec=ChevronHeraldedExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide_heralded.md"),
    analysis=Analysis(fn=_analyze),
)
