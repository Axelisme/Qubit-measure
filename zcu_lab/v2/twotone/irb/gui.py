"""Measure GUI attachment for paired reference/interleaved randomized benchmarking."""

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

from zcu_lab.v2.twotone.irb.core import IRB_Exp, IRBCfg, IRBResult

from .form import make_form

IRBRunResult: TypeAlias = RunRecord[IRBCfg, IRBResult]


@dataclass
class IRBAnalyzeResult(AnalyzeResultBase):
    """Unitless decay/error estimates, statistical confidence and pairing counts."""

    p_reference: float
    p_interleaved: float
    reference_epc: float
    gate_error: float
    gate_fidelity: float
    fidelity_ci_low: float
    fidelity_ci_high: float
    n_paired_seeds: int
    n_paired_rounds: int
    bootstrap_successes: int
    warning: str


class IRBAdapter(BaseAdapter[IRBCfg, IRBRunResult, IRBAnalyzeResult]):
    @staticmethod
    def make_form(ctx: FormContext) -> Form[IRBCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    """Run, persist and fit RB using calibrated X90/X180 and readout modules."""

    experiment_tag: ClassVar[str | None] = IRB_Exp.persistence_tag()

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
    ) -> IRBAnalyzeResult:
        """Fit the saved run's mean IQ decay and publish its fit figure."""
        result = IRB_Exp().analyze(req.run_result, None, plots=plots)
        return IRBAnalyzeResult(
            p_reference=result.p_reference,
            p_interleaved=result.p_interleaved,
            reference_epc=result.reference_epc,
            gate_error=result.gate_error,
            gate_fidelity=result.gate_fidelity,
            fidelity_ci_low=result.fidelity_ci_low,
            fidelity_ci_high=result.fidelity_ci_high,
            n_paired_seeds=result.n_paired_seeds,
            n_paired_rounds=result.n_paired_rounds,
            bootstrap_successes=result.bootstrap_successes,
            warning=result.warning,
        )
