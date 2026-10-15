from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

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

from zcu_lab.v2.lookback.core import (
    LookbackAnalyzeOptions,
    LookbackCfg,
    LookbackExp,
    LookbackResult,
)
from zcu_lab.v2.lookback.form import make_form

LookbackRunResult: TypeAlias = RunRecord[LookbackCfg, LookbackResult]


@dataclass
class LookbackAnalyzeParams:
    ratio: Annotated[float, ParamMeta(label="Threshold ratio", decimals=3)] = 0.1
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=3)] = 1.0
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")] = True


@dataclass
class LookbackAnalyzeResult(AnalyzeResultBase):
    predict_offset: float


class LookbackAdapter(
    BaseAdapter[
        LookbackCfg,
        LookbackRunResult,
        LookbackAnalyzeResult,
        LookbackAnalyzeParams,
    ]
):
    experiment_tag: ClassVar[str | None] = LookbackExp.persistence_tag()
    exp_cls = LookbackExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[LookbackCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: LookbackCfg,
        *,
        context: RunContext,
    ) -> LookbackRunResult:
        del req
        result = LookbackExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[LookbackRunResult, LookbackAnalyzeParams],
        *,
        plots: Plots,
    ) -> LookbackAnalyzeResult:
        params = req.analyze_params
        answer = LookbackExp().analyze(
            req.run_result,
            LookbackAnalyzeOptions(
                ratio=params.ratio,
                smooth=params.smooth,
                plot_fit=params.plot_fit,
            ),
            plots=plots,
        )
        return LookbackAnalyzeResult(predict_offset=answer.predict_offset)

    def get_writeback_items(
        self, req: WritebackRequest[LookbackRunResult, LookbackAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        """Propose detected arrival time in us at captured resonator.ext.time_of_fly.

        req supplies the analysis and its capture; source.cfg may be None.
        WritebackService resolves the destination on req.ctx.parameters and
        rejects missing capture or unbound roles. No live focus lookup or write
        occurs here.
        """
        return [
            ParameterWriteback(
                target_name="$resonator.ext.time_of_fly",
                description="Readout trigger offset prediction (us)",
                proposed_value=req.analyze_result.predict_offset,
            )
        ]
