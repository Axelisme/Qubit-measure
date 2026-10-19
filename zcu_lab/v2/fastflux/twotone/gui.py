"""Measure GUI adapter for flux-pulsed qubit spectroscopy."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import TwotoneCfg, TwoToneExp, TwoToneResult
from .form import make_form

TwoToneRunResult: TypeAlias = RunRecord[TwotoneCfg, TwoToneResult]


@dataclass
class TwoToneAnalyzeResult(AnalyzeResultBase):
    """The projected spectroscopy map has no fitted scalars or writeback."""


class TwoToneAdapter(
    BaseAdapter[TwotoneCfg, TwoToneRunResult, TwoToneAnalyzeResult, NoAnalyzeParams]
):
    """Acquire and display frequency versus flux-pulse gain without a model fit."""

    exp_cls = TwoToneExp
    experiment_tag: ClassVar[str | None] = TwoToneExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[TwotoneCfg]:
        """Build this tab from the captured Open/Reset sources."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: TwotoneCfg, *, context: RunContext
    ) -> TwoToneRunResult:
        del req
        return RunRecord(cfg=cfg, result=TwoToneExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[TwoToneRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> TwoToneAnalyzeResult:
        TwoToneExp().analyze(req.run_result, None, plots=plots)
        return TwoToneAnalyzeResult()
