"""Measure GUI attachment for single-qubit Clifford randomized benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.rb.core import RB_Exp, RB_Result, RBCfg

from .form import make_form

RBRunResult: TypeAlias = RunRecord[RBCfg, RB_Result]


@dataclass
class RBAnalyzeResult(AnalyzeResultBase):
    """Single-qubit error per Clifford and mean Clifford fidelity (unitless)."""

    epc: float
    fidelity: float


class RBAdapter(BaseAdapter[RBCfg, RBRunResult, RBAnalyzeResult]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[RBCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    """Run, persist and fit RB using calibrated X90/X180 and readout modules."""

    exp_cls = RB_Exp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: RBCfg,
        *,
        context: RunContext,
    ) -> RBRunResult:
        """Execute the accepted cfg; propagate acquisition failures to the GUI."""
        del req
        return RunRecord(cfg=cfg, result=RB_Exp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[RBRunResult, None], *, plots: Plots
    ) -> RBAnalyzeResult:
        """Fit the saved run's mean IQ decay and publish its fit figure."""
        result = RB_Exp().analyze(req.run_result, None, plots=plots)
        return RBAnalyzeResult(epc=result.epc, fidelity=result.fidelity)
