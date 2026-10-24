"""Measure GUI attachment for paired reference/interleaved randomized benchmarking."""

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

from zcu_lab.v2.twotone.irb.analysis import IRBAnalysis
from zcu_lab.v2.twotone.irb.core import IRB_Exp, IRBCfg, IRBResult

from .form import make_form

IRBRunResult: TypeAlias = RunRecord[IRBCfg, IRBResult]


class IRBAdapter(BaseAdapter[IRBCfg, IRBRunResult, IRBAnalysis]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[IRBCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    """Run, persist and fit RB using calibrated X90/X180 and readout modules."""

    exp_cls = IRB_Exp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: IRBCfg,
        *,
        context: RunContext,
    ) -> IRBRunResult:
        """Execute the accepted cfg; propagate acquisition failures to the GUI."""
        del req
        return RunRecord(cfg=cfg, result=IRB_Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[IRBRunResult, None], *, plots: Plots
    ) -> IRBAnalysis:
        """Fit the saved run's mean IQ decay and publish its fit figure."""
        return IRB_Exp().analyze(req.run_result, None, plots=plots)
