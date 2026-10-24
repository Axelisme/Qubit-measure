from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.reset_check.core import (
    ResetCheckAnalysis,
    ResetCheckAnalyzeOptions,
    ResetCheckCfg,
    ResetCheckExp,
    ResetCheckResult,
)
from zcu_lab.v2.singleshot.reset_check.form import make_form

SsResetCheckRunResult = RunRecord[ResetCheckCfg, ResetCheckResult]


class SsResetCheckAdapter(
    BaseAdapter[ResetCheckCfg, SsResetCheckRunResult, ResetCheckAnalysis]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[ResetCheckCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = ResetCheckExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: ResetCheckCfg,
        *,
        context: RunContext,
    ) -> SsResetCheckRunResult:
        del req
        result = ResetCheckExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[SsResetCheckRunResult, None],
        *,
        plots: Plots,
    ) -> ResetCheckAnalysis:
        return ResetCheckExp().analyze(
            req.run_result,
            ResetCheckAnalyzeOptions(
                confusion_matrix=read_confusion_matrix(req.parameters)
            ),
            plots=plots,
        )
