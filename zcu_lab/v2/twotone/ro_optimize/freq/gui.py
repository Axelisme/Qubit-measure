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

from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.ro_optimize.freq.form import make_form

RoOptFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class RoOptFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 2.0


@dataclass
class RoOptFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float


class RoOptFreqAdapter(
    BaseAdapter[
        FreqCfg,
        RoOptFreqRunResult,
        RoOptFreqAnalyzeResult,
        RoOptFreqAnalyzeParams,
    ]
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
    ) -> RoOptFreqRunResult:
        del req
        return RunRecord(cfg, FreqExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[RoOptFreqRunResult, RoOptFreqAnalyzeParams],
        *,
        plots: Plots,
    ) -> RoOptFreqAnalyzeResult:
        params = req.analyze_params
        options = FreqAnalyzeOptions(
            smooth=params.smooth,
            smooth_method=params.smooth_method,
        )
        result = FreqExp().analyze(req.run_result, options, plots=plots)
        return RoOptFreqAnalyzeResult(
            best_freq=result.best_freq,
        )

    def get_writeback_items(
        self, req: WritebackRequest[RoOptFreqRunResult, RoOptFreqAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        items: list[WritebackItem] = [
            ParameterWriteback(
                target_name="$qubit.ext.best_ro_freq",
                description="Optimal readout frequency (MHz)",
                proposed_value=result.best_freq,
            ),
        ]
        items.extend(
            readout_dpm_writeback_items(
                req.ctx,
                req.run_result.cfg,
                proposed={"best_ro_freq": result.best_freq},
            )
        )
        return items
