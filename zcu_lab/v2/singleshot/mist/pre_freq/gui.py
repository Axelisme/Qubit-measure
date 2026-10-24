"""Measure GUI adapter for the MIST preparation-frequency sweep."""

from __future__ import annotations

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

from zcu_lab.v2.singleshot.mist.pre_freq.core import (
    PreFreqAnalyzeOptions,
    PreFreqCfg,
    PreFreqExp,
    PreFreqResult,
)
from zcu_lab.v2.singleshot.mist.pre_freq.form import make_form
from zcu_lab.v2.support.measure.singleshot_shared import read_confusion_matrix

MistPreFreqRunResult: TypeAlias = RunRecord[PreFreqCfg, PreFreqResult]


def _analyze(
    req: AnalyzeRequest[MistPreFreqRunResult, None],
    *,
    plots: Plots,
) -> None:
    """Plot G/E/Other, using optional confusion correction from the run capture."""
    options = PreFreqAnalyzeOptions(
        confusion_matrix=read_confusion_matrix(req.parameters)
    )
    PreFreqExp().analyze(req.run_result, options, plots=plots)
    return


def _run(
    cfg: PreFreqCfg, *, context: RunContext
) -> RunRecord[PreFreqCfg, PreFreqResult]:
    return RunRecord(cfg=cfg, result=PreFreqExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=PreFreqExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
