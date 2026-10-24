from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.dual_tone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.reset.dual_tone.freq.form import make_form

DualToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class DualToneFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class DualToneFreqAnalyzeResult(AnalyzeResultBase):
    freq1: float
    freq2: float


class DualToneFreqAdapter(
    BaseAdapter[
        FreqCfg,
        DualToneFreqRunResult,
        DualToneFreqAnalyzeResult,
        DualToneFreqAnalyzeParams,
    ]
):
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
    ) -> DualToneFreqRunResult:
        del req
        cfg = cfg.with_updates(method="hard")
        result = FreqExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[DualToneFreqRunResult, DualToneFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> DualToneFreqAnalyzeResult:
        params = req.analyze_params
        result = FreqExp().analyze(
            req.run_result,
            FreqAnalyzeOptions(
                smooth=params.smooth, smooth_method=params.smooth_method
            ),
            plots=plots,
        )
        return DualToneFreqAnalyzeResult(freq1=result.freq1, freq2=result.freq2)

    def get_writeback_items(
        self,
        req: WritebackRequest[DualToneFreqRunResult, DualToneFreqAnalyzeResult],
    ) -> Sequence[WritebackItem]:
        """Propose this analysis's two frequencies to captured qubit ext."""
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_f1",
                description="Dual-tone reset frequency 1 (MHz)",
                proposed_value=result.freq1,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_f2",
                description="Dual-tone reset frequency 2 (MHz)",
                proposed_value=result.freq2,
            ),
        ]
        return items
