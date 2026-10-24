"""AllXY gate-error check adapter.

Runs the 21 standard AllXY gate pairs and reuses ``AllXY_Exp.analyze``, which
fits power and detuning errors; the summary reports them.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.allxy.core import (
    AllXY_Exp,
    AllXY_Result,
    AllXYAnalysis,
    AllXYAnalyzeOptions,
    AllXYCfg,
)

from .form import make_form

AllXYRunResult: TypeAlias = RunRecord[AllXYCfg, AllXY_Result]


class AllXYAdapter(
    BaseAdapter[AllXYCfg, AllXYRunResult, AllXYAnalysis, AllXYAnalyzeOptions]
):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[AllXYCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

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
        req: AnalyzeRequest[AllXYRunResult, AllXYAnalyzeOptions],
        *,
        plots: Plots,
    ) -> AllXYAnalysis:
        return AllXY_Exp().analyze(
            req.run_result,
            req.analyze_params,
            plots=plots,
        )
