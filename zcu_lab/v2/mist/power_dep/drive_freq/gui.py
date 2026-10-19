"""Measure GUI attachment for the native MIST frequency/gain experiment."""

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

from .core import DriveFreqCfg, DriveFreqExp, DriveFreqResult
from .form import make_form

DriveFreqRunResult: TypeAlias = RunRecord[DriveFreqCfg, DriveFreqResult]


@dataclass
class DriveFreqAnalyzeResult(AnalyzeResultBase):
    """The baseline-subtracted map has no fitted scalars or writeback."""


class DriveFreqAdapter(
    BaseAdapter[
        DriveFreqCfg, DriveFreqRunResult, DriveFreqAnalyzeResult, NoAnalyzeParams
    ]
):
    """Acquire native complex IQ and publish the core's MIST response map."""

    exp_cls = DriveFreqExp
    experiment_tag: ClassVar[str | None] = DriveFreqExp.persistence_tag()
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True, requires_soc=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[DriveFreqCfg]:
        """Build a fresh map form using the fixed Open/Reset source capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: DriveFreqCfg, *, context: RunContext
    ) -> DriveFreqRunResult:
        """Reset, optionally initialize, probe and read out using the displayed cfg."""
        del req
        return RunRecord(cfg, DriveFreqExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[DriveFreqRunResult, NoAnalyzeParams],
        *,
        plots: Plots,
    ) -> DriveFreqAnalyzeResult:
        """Publish the saved source's baseline-subtracted magnitude as `fit`."""
        DriveFreqExp().analyze(req.run_result, None, plots=plots)
        return DriveFreqAnalyzeResult()
