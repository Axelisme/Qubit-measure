"""Measure and display flux-transient phase tomography in the measure GUI."""

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

from .core import PhaseAnalysis, PhaseCfg, PhaseExp, PhaseResult
from .form import make_form

PhaseRunResult: TypeAlias = RunRecord[PhaseCfg, PhaseResult]


class PhaseAdapter(BaseAdapter[PhaseCfg, PhaseRunResult, PhaseAnalysis]):
    exp_cls = PhaseExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PhaseCfg]:
        """Build pulse choices and scan inputs from captured Open/Reset sources."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: PhaseCfg, *, context: RunContext
    ) -> PhaseRunResult:
        """Move the π/2 pair through the flux pulse and keep this run's cfg."""
        del req
        return RunRecord(cfg=cfg, result=PhaseExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[PhaseRunResult, None],
        *,
        plots: Plots,
    ) -> PhaseAnalysis:
        """Plot tomography and unwrapped phase using only the source run timing."""
        return PhaseExp().analyze(req.run_result, None, plots=plots)
