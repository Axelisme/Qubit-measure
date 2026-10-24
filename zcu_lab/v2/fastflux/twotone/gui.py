"""Measure GUI adapter for flux-pulsed qubit spectroscopy."""

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from .core import TwotoneCfg, TwoToneExp, TwoToneResult
from .form import make_form

TwoToneRunResult: TypeAlias = RunRecord[TwotoneCfg, TwoToneResult]


def _analyze(req: AnalyzeRequest[TwoToneRunResult, None], *, plots: Plots) -> None:
    TwoToneExp().analyze(req.run_result, None, plots=plots)
    return


def run(
    cfg: TwotoneCfg, *, context: RunContext
) -> RunRecord[TwotoneCfg, TwoToneResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=TwoToneExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=TwoToneExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
