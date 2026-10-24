from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.reset_check.core import (
    ResetCheckAnalyzeOptions,
    ResetCheckCfg,
    ResetCheckExp,
    ResetCheckResult,
)
from zcu_lab.v2.singleshot.reset_check.form import make_form

SsResetCheckRunResult = RunRecord[ResetCheckCfg, ResetCheckResult]


@dataclass
class SsResetCheckAnalyzeResult(AnalyzeResultBase):
    reset_mean_excited_population: float
    reset_max_excited_population: float
    reset_max_other_population: float
    worst_sample_gain: float
    analyzed_reset_points: int


def _analyze(
    req: AnalyzeRequest[SsResetCheckRunResult, None],
    *,
    plots: Plots,
) -> SsResetCheckAnalyzeResult:
    analysis = ResetCheckExp().analyze(
        req.run_result,
        ResetCheckAnalyzeOptions(
            confusion_matrix=read_confusion_matrix(req.parameters)
        ),
        plots=plots,
    )
    return SsResetCheckAnalyzeResult(
        analysis.reset_mean_excited_population,
        analysis.reset_max_excited_population,
        analysis.reset_max_other_population,
        analysis.worst_sample_gain,
        analysis.analyzed_reset_points,
    )


def _run(
    cfg: ResetCheckCfg, *, context: RunContext
) -> RunRecord[ResetCheckCfg, ResetCheckResult]:
    return RunRecord(cfg=cfg, result=ResetCheckExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=ResetCheckExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
