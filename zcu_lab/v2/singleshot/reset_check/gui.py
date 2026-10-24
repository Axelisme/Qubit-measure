from __future__ import annotations

from pathlib import Path

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import Adapter, Analysis, AnalyzeRequest
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.singleshot.reset_check.core import (
    ResetCheckAnalysis,
    ResetCheckAnalyzeOptions,
    ResetCheckCfg,
    ResetCheckExp,
    ResetCheckResult,
)
from zcu_lab.v2.singleshot.reset_check.form import make_form
from zcu_lab.v2.support.measure.singleshot_shared import read_confusion_matrix

SsResetCheckRunResult = RunRecord[ResetCheckCfg, ResetCheckResult]


def _analyze(
    req: AnalyzeRequest[SsResetCheckRunResult, None], *, plots: Plots
) -> ResetCheckAnalysis:
    return ResetCheckExp().analyze(
        req.run_result,
        ResetCheckAnalyzeOptions(
            confusion_matrix=read_confusion_matrix(req.parameters)
        ),
        plots=plots,
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
