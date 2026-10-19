"""Measure GUI adapter for fast-flux transient spectroscopy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    NoAnalyzeParams,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.fastflux.distortion.freq.core import FreqCfg, FreqExp, FreqResult
from zcu_lab.v2.fastflux.distortion.freq.form import make_form

FreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class FreqAnalyzeResult(AnalyzeResultBase):
    """The resonance trace is published in the fit figure, without writeback."""


class FastFluxFreqAdapter(
    BaseAdapter[FreqCfg, FreqRunResult, FreqAnalyzeResult, NoAnalyzeParams]
):
    experiment_tag: ClassVar[str | None] = FreqExp.persistence_tag()
    exp_cls = FreqExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FreqCfg]:
        """Build a fresh form from captured sources."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: FreqCfg, *, context: RunContext
    ) -> FreqRunResult:
        del req
        return RunRecord(cfg, FreqExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[FreqRunResult, NoAnalyzeParams], *, plots: Plots
    ) -> FreqAnalyzeResult:
        FreqExp().analyze(req.run_result, None, plots=plots)
        return FreqAnalyzeResult()
