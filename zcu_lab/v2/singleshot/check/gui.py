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

from zcu_lab.v2.singleshot.check.core import (
    CheckAnalyzeOptions,
    CheckCfg,
    CheckExp,
    CheckResult,
)
from zcu_lab.v2.singleshot.check.form import make_form

CheckRunResult: TypeAlias = RunRecord[CheckCfg, CheckResult]


def _analyze(req: AnalyzeRequest[CheckRunResult, None], *, plots: Plots) -> None:
    CheckExp().analyze(req.run_result, CheckAnalyzeOptions(), plots=plots)


def _run(cfg: CheckCfg, *, context: RunContext) -> RunRecord[CheckCfg, CheckResult]:
    return RunRecord(cfg=cfg, result=CheckExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=CheckExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
