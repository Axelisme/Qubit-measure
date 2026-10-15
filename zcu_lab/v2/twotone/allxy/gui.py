"""AllXY gate-error check adapter.

Runs the 21 standard AllXY gate pairs and reuses ``AllXY_Exp.analyze``, which
fits power and detuning errors; the summary reports them.
"""

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

from zcu_lab.v2.twotone.allxy.core import (
    AllXY_Exp,
    AllXY_Result,
    AllXYAnalyzeOptions,
    AllXYCfg,
)

from .form import make_form

AllXYRunResult: TypeAlias = RunRecord[AllXYCfg, AllXY_Result]


@dataclass
class AllXYAnalyzeParams:
    fit_ge: Annotated[bool, ParamMeta(label="Fit g/e levels")] = False


@dataclass
class AllXYAnalyzeResult(AnalyzeResultBase):
    amplitude_error: float
    detune_param: float
    power_err: float
    detune_err: float
    residual_rms: float


class AllXYAdapter(
    BaseAdapter[AllXYCfg, AllXYRunResult, AllXYAnalyzeResult, AllXYAnalyzeParams]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[AllXYCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    experiment_tag: ClassVar[str | None] = AllXY_Exp.persistence_tag()
    exp_cls = AllXY_Exp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: AllXYCfg,
        *,
        context: RunContext,
    ) -> AllXYRunResult:
        del req
        return RunRecord(cfg=cfg, result=AllXY_Exp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[AllXYRunResult, AllXYAnalyzeParams],
        *,
        plots: Plots,
    ) -> AllXYAnalyzeResult:
        analysis = AllXY_Exp().analyze(
            req.run_result,
            AllXYAnalyzeOptions(fit_ge=req.analyze_params.fit_ge),
            plots=plots,
        )
        return AllXYAnalyzeResult(
            amplitude_error=analysis.amplitude_error,
            detune_param=analysis.detune_param,
            power_err=analysis.power_err,
            detune_err=analysis.detune_err,
            residual_rms=analysis.residual_rms,
        )
