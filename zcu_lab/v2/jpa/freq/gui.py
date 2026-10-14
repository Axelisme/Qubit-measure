"""jpa/freq GUI adapter — JPA pump-frequency tracer.

Owns the jpa/freq cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.freq.core import FreqCfg, FreqExp, FreqResult
from zcu_lab.v2.jpa.freq.form import make_form

JpaFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class JpaFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float


class JpaFreqAdapter(
    BaseAdapter[FreqCfg, JpaFreqRunResult, JpaFreqAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: FreqCfg,
        *,
        context: RunContext,
    ) -> JpaFreqRunResult:
        del req
        return RunRecord(cfg, FreqExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaFreqRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> JpaFreqAnalyzeResult:
        answer = FreqExp().analyze(req.run_result, None, plots=plots)
        return JpaFreqAnalyzeResult(best_freq=answer.best_freq)

    def get_writeback_items(
        self, req: WritebackRequest[JpaFreqRunResult, JpaFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            ParameterWriteback(
                target_name="$jpa.pump_freq",
                description="Best JPA pump frequency (MHz)",
                proposed_value=req.analyze_result.best_freq,
            )
        ]

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.qub_name}_jpa_freq_{time.strftime('%m%d')}"
