"""GUI adapters for gain and frequency scans of the ZigZag experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.zigzag_sweep.core import (
    ZigZagScanAnalyzeOptions,
    ZigZagScanCfg,
    ZigZagScanExp,
    ZigZagScanResult,
)

from .form import make_freq_form, make_gain_form

ZigZagScanRunResult: TypeAlias = RunRecord[ZigZagScanCfg, ZigZagScanResult]


@dataclass
class ZigZagScanAnalyzeParams:
    find_min: Annotated[float | None, ParamMeta(label="Search from")] = None
    find_max: Annotated[float | None, ParamMeta(label="Search to")] = None


@dataclass
class ZigZagScanAnalyzeResult(AnalyzeResultBase):
    best_value: float


class _ZigZagScanBase(
    BaseAdapter[
        ZigZagScanCfg,
        ZigZagScanRunResult,
        ZigZagScanAnalyzeResult,
        ZigZagScanAnalyzeParams,
    ]
):
    """Shared body; subclasses choose the single swept parameter."""

    exp_cls = ZigZagScanExp

    def run(
        self,
        req: RunRequest,
        cfg: ZigZagScanCfg,
        *,
        context: RunContext,
    ) -> ZigZagScanRunResult:
        del req
        result = ZigZagScanExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self,
        req: AnalyzeRequest[ZigZagScanRunResult, ZigZagScanAnalyzeParams],
        *,
        plots: Plots,
    ) -> ZigZagScanAnalyzeResult:
        params = req.analyze_params
        analysis = ZigZagScanExp().analyze(
            req.run_result,
            ZigZagScanAnalyzeOptions(find_range=(params.find_min, params.find_max)),
            plots=plots,
        )
        return ZigZagScanAnalyzeResult(best_value=analysis.min_value)


class ZigZagScanGainAdapter(_ZigZagScanBase):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[ZigZagScanCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_gain_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.gain.md")


class ZigZagScanFreqAdapter(_ZigZagScanBase):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[ZigZagScanCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_freq_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.freq.md")
