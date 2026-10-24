"""Measure GUI attachment for the native MIST frequency/gain experiment."""

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import DriveFreqCfg, DriveFreqExp, DriveFreqResult
from .form import make_form

DriveFreqRunResult: TypeAlias = RunRecord[DriveFreqCfg, DriveFreqResult]


class DriveFreqAdapter(BaseAdapter[DriveFreqCfg, DriveFreqRunResult]):
    """Acquire native complex IQ and publish the core's MIST response map."""

    exp_cls = DriveFreqExp
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
        req: AnalyzeRequest[DriveFreqRunResult, None],
        *,
        plots: Plots,
    ) -> None:
        """Publish the saved source's baseline-subtracted magnitude as `fit`."""
        DriveFreqExp().analyze(req.run_result, None, plots=plots)
        return
