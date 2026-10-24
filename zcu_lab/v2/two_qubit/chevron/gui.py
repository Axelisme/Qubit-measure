"""Measure adapter for chevron."""

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

from .._rabi_scan import RabiAnalysis, RabiScanResult
from .core import ChevronCfg, ChevronExp
from .form import make_form

ChevronRun: TypeAlias = RunRecord[ChevronCfg, RabiScanResult]


class ChevronAdapter(BaseAdapter[ChevronCfg, ChevronRun, RabiAnalysis]):
    """Fixed-capture acquisition, saved-source analysis and native restore."""

    exp_cls = ChevronExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[ChevronCfg]:
        """Capture editable pulses, GE centers and locked role/tone identities."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: ChevronCfg, *, context: RunContext
    ) -> ChevronRun:
        """Pair original cfg with full raw; caller checks cancel_signal errors."""
        del req
        return RunRecord(cfg=cfg, result=ChevronExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[ChevronRun, None], *, plots: Plots
    ) -> RabiAnalysis:
        """Analyze only saved source cfg/raw, never the current Form or library."""
        return ChevronExp().analyze(req.run_result, None, plots=plots)
