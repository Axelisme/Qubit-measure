"""Measure GUI attachment for the native MIST frequency/gain experiment."""

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

from .core import DriveFreqCfg, DriveFreqExp, DriveFreqResult
from .form import make_form

DriveFreqRunResult: TypeAlias = RunRecord[DriveFreqCfg, DriveFreqResult]


def _analyze(
    req: AnalyzeRequest[DriveFreqRunResult, None],
    *,
    plots: Plots,
) -> None:
    """Publish the saved source's baseline-subtracted magnitude as `fit`."""
    DriveFreqExp().analyze(req.run_result, None, plots=plots)
    return


def run(
    cfg: DriveFreqCfg, *, context: RunContext
) -> RunRecord[DriveFreqCfg, DriveFreqResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=DriveFreqExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=DriveFreqExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
