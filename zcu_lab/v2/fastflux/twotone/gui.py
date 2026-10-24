"""Measure GUI adapter for flux-pulsed qubit spectroscopy."""

from pathlib import Path
from typing import TypeAlias

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


DEFINITION = Adapter(
    spec=TwoToneExp.AXES_SPEC,
    make_form=make_form,
    run=TwoToneExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
