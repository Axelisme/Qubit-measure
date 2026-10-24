"""Measure adapter for four thermal branches of two-line spectroscopy."""

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

from .analysis import SpectroscopyAnalysis
from .core import SpectroscopyCfg, SpectroscopyExp, SpectroscopyResult
from .form import make_form

SpectroscopyRun: TypeAlias = RunRecord[SpectroscopyCfg, SpectroscopyResult]


class SpectroscopyAdapter(
    BaseAdapter[SpectroscopyCfg, SpectroscopyRun, SpectroscopyAnalysis]
):
    """Fixed-capture Form, full raw acquisition, diagnostic analysis and restore."""

    exp_cls = SpectroscopyExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[SpectroscopyCfg]:
        """Resolve role/tone identities and editable pulses from one Open/Reset capture."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: SpectroscopyCfg, *, context: RunContext
    ) -> SpectroscopyRun:
        """Keep original cfg with raw; caller checks cancel_signal.raise_if_error()."""
        del req
        return RunRecord(cfg=cfg, result=SpectroscopyExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[SpectroscopyRun, None], *, plots: Plots
    ) -> SpectroscopyAnalysis:
        """Use only source cfg/raw; never read current parameters or propose peaks."""
        return SpectroscopyExp().analyze(req.run_result, None, plots=plots)
