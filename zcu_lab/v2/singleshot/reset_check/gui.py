from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
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


class SsResetCheckAdapter(
    BaseAdapter[
        ResetCheckCfg, SsResetCheckRunResult, SsResetCheckAnalyzeResult, NoAnalyzeParams
    ]
):
    experiment_tag: ClassVar[str | None] = ResetCheckExp.persistence_tag()

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
        req: AnalyzeRequest[SsResetCheckRunResult, NoAnalyzeParams],
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
