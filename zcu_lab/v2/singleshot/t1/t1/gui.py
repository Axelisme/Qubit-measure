from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.t1.t1.core import T1AnalyzeOptions, T1Cfg, T1Exp, T1Result
from zcu_lab.v2.singleshot.t1.t1.form import make_form

# Transition-rate analysis publishes figures without numeric writeback.
SsT1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass
class SsT1AnalyzeResult(AnalyzeResultBase):
    pass


class SsT1Adapter(
    BaseAdapter[T1Cfg, SsT1RunResult, SsT1AnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = T1Exp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1Cfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = T1Exp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: T1Cfg,
        *,
        context: RunContext,
    ) -> SsT1RunResult:
        del req
        return RunRecord(cfg, T1Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SsT1RunResult, NoAnalyzeParams], *, plots: Plots
    ) -> SsT1AnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        # ``skip`` is not exposed as a user knob — users can re-run with a shorter
        # sweep instead.
        confusion = read_confusion_matrix(req.parameters)
        T1Exp().analyze(
            req.run_result, T1AnalyzeOptions(confusion_matrix=confusion), plots=plots
        )
        return SsT1AnalyzeResult()

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_ss_t1_{time.strftime('%m%d')}"
