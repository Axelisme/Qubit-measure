"""Measure GUI adapter for fast-flux transient spectroscopy."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.fastflux.distortion.freq.core import FreqCfg, FreqExp, FreqResult
from zcu_lab.v2.fastflux.distortion.freq.form import make_form

FreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


def _analyze(req: AnalyzeRequest[FreqRunResult, None], *, plots: Plots) -> None:
    FreqExp().analyze(req.run_result, None, plots=plots)
    return


DEFINITION = Adapter(
    spec=FreqExp.AXES_SPEC,
    make_form=make_form,
    run=FreqExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
