from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.reset.bath.freq.core import (
    FreqGainAnalysis,
    FreqGainAnalyzeOptions,
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_lab.v2.twotone.reset.bath.freq.form import make_form

BathFreqGainRunResult: TypeAlias = RunRecord[FreqGainCfg, FreqGainResult]


@dataclass
class BathFreqGainAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


class BathFreqGainAdapter(
    BaseAdapter[
        FreqGainCfg,
        BathFreqGainRunResult,
        FreqGainAnalysis,
        BathFreqGainAnalyzeParams,
    ]
):
    exp_cls = FreqGainExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqGainCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: FreqGainCfg,
        *,
        context: RunContext,
    ) -> BathFreqGainRunResult:
        del req
        result = FreqGainExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[BathFreqGainRunResult, BathFreqGainAnalyzeParams],
        *,
        plots: Plots,
    ) -> FreqGainAnalysis:
        params = req.analyze_params
        return FreqGainExp().analyze(
            req.run_result,
            FreqGainAnalyzeOptions(
                smooth=params.smooth, smooth_method=params.smooth_method
            ),
            plots=plots,
        )

    def get_writeback_items(
        self,
        req: WritebackRequest[BathFreqGainRunResult, FreqGainAnalysis],
    ) -> Sequence[WritebackItem]:
        """Propose this analysis's cavity gain then frequency to qubit ext."""
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.reset_bath_gain",
                description="Bath-reset cavity gain (a.u.)",
                proposed_value=result.gain,
            ),
            ParameterWriteback(
                target_name="$qubit.ext.reset_bath_freq",
                description="Bath-reset cavity frequency (MHz)",
                proposed_value=result.freq,
            ),
        ]
        return items
