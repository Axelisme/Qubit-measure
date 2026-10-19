"""Measure GUI adapter for the MIST preparation-frequency sweep."""

from __future__ import annotations

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

from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.mist.pre_freq.core import (
    PreFreqAnalyzeOptions,
    PreFreqCfg,
    PreFreqExp,
    PreFreqResult,
)
from zcu_lab.v2.singleshot.mist.pre_freq.form import make_form

MistPreFreqRunResult: TypeAlias = RunRecord[PreFreqCfg, PreFreqResult]


@dataclass
class MistPreFreqAnalyzeResult(AnalyzeResultBase):
    """Population curves are published as a fit figure, without scalar estimates."""


class MistPreFreqAdapter(
    BaseAdapter[
        PreFreqCfg, MistPreFreqRunResult, MistPreFreqAnalyzeResult, NoAnalyzeParams
    ]
):
    """Expose preparation-frequency acquisition and population analysis in Measure."""

    experiment_tag: ClassVar[str | None] = PreFreqExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    exp_cls = PreFreqExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[PreFreqCfg]:
        """Build this adapter's form from the captured Open/Reset sources."""
        return make_form(ctx)

    def run(
        self,
        req: RunRequest,
        cfg: PreFreqCfg,
        *,
        context: RunContext,
    ) -> MistPreFreqRunResult:
        """Sweep preparation frequency and pair G/E fractions with their source cfg."""
        del req
        return RunRecord(cfg, PreFreqExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[MistPreFreqRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> MistPreFreqAnalyzeResult:
        """Plot G/E/Other, using optional confusion correction from the run capture."""
        options = PreFreqAnalyzeOptions(
            confusion_matrix=read_confusion_matrix(req.parameters)
        )
        PreFreqExp().analyze(req.run_result, options, plots=plots)
        return MistPreFreqAnalyzeResult()
