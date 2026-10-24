from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone.core import (
    T1WithToneAnalyzeOptions,
    T1WithToneCfg,
    T1WithToneExp,
    T1WithToneResult,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone.form import make_form

# The numeric t1 is written to the t1_with_tone parameter context key.
SsT1ToneRunResult: TypeAlias = RunRecord[T1WithToneCfg, T1WithToneResult]


@dataclass
class SsT1ToneAnalyzeResult(AnalyzeResultBase):
    t1: float
    t1_b: float


class SsT1ToneAdapter(
    BaseAdapter[T1WithToneCfg, SsT1ToneRunResult, SsT1ToneAnalyzeResult]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[T1WithToneCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = T1WithToneExp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: T1WithToneCfg,
        *,
        context: RunContext,
    ) -> SsT1ToneRunResult:
        del req
        return RunRecord(cfg, T1WithToneExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SsT1ToneRunResult, None], *, plots: Plots
    ) -> SsT1ToneAnalyzeResult:
        # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
        confusion = read_confusion_matrix(req.parameters)
        result = T1WithToneExp().analyze(
            req.run_result,
            T1WithToneAnalyzeOptions(confusion_matrix=confusion),
            plots=plots,
        )
        return SsT1ToneAnalyzeResult(t1=result.t1, t1_b=result.t1_b)

    def get_writeback_items(
        self,
        req: WritebackRequest[SsT1ToneRunResult, SsT1ToneAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        # Key ``t1_with_tone`` per single_qubit.md:3079.
        return [
            ParameterWriteback(
                target_name="$qubit.ext.t1_with_tone",
                description="T1 with probe tone (us)",
                proposed_value=req.analyze_result.t1,
            ),
        ]
