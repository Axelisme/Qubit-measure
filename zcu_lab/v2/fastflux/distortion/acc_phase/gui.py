"""Measure GUI attachment for accumulated-phase flux distortion."""

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

from .core import AccPhaseCfg, AccPhaseExp, AccPhaseResult
from .form import make_form

AccPhaseRunResult: TypeAlias = RunRecord[AccPhaseCfg, AccPhaseResult]


class AccPhaseAdapter(BaseAdapter[AccPhaseCfg, AccPhaseRunResult]):
    experiment_tag: ClassVar[str | None] = AccPhaseExp.persistence_tag()
    exp_cls = AccPhaseExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[AccPhaseCfg]:
        """Build inputs from the sources captured at Open or Reset."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: AccPhaseCfg, *, context: RunContext
    ) -> AccPhaseRunResult:
        """Keep the acquired IQ paired with its actual timing and pulse cfg."""
        del req
        return RunRecord(cfg, AccPhaseExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[AccPhaseRunResult, None],
        *,
        plots: Plots,
    ) -> None:
        """Plot phase projections and derived detuning from the source record."""
        AccPhaseExp().analyze(req.run_result, None, plots=plots)
        return
