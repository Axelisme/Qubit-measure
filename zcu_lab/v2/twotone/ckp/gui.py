"""CKP GUI adapter for ground/excited-resolved two-tone spectroscopy."""

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

from zcu_lab.v2.twotone.ckp.core import CKP_Cfg, CKP_Exp, CKP_Result

from .form import make_form

CKPRunResult: TypeAlias = RunRecord[CKP_Cfg, CKP_Result]


@dataclass
class CKPAnalyzeResult(AnalyzeResultBase):
    chi: float
    kappa: float
    res_freq: float


class CKPAdapter(BaseAdapter[CKP_Cfg, CKPRunResult, CKPAnalyzeResult]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[CKP_Cfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = CKP_Exp

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: CKP_Cfg,
        *,
        context: RunContext,
    ) -> CKPRunResult:
        del req
        result = CKP_Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[CKPRunResult, None], *, plots: Plots
    ) -> CKPAnalyzeResult:
        analysis = CKP_Exp().analyze(req.run_result, None, plots=plots)
        return CKPAnalyzeResult(
            chi=analysis.chi,
            kappa=analysis.kappa,
            res_freq=analysis.res_freq,
        )

    def get_writeback_items(
        self, req: WritebackRequest[CKPRunResult, CKPAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            ParameterWriteback(
                target_name="$qubit.ext.chi",
                description="Dispersive shift chi (MHz)",
                proposed_value=result.chi,
            ),
            ParameterWriteback(
                target_name="$resonator.kappa",
                description="Resonator linewidth kappa (MHz)",
                proposed_value=result.kappa,
            ),
            ParameterWriteback(
                target_name="$resonator.freq",
                description="Readout frequency (MHz)",
                proposed_value=result.res_freq,
            ),
        ]
