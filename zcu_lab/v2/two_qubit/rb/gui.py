"""Measure adapter for source-bound paired two-qubit RB."""

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

from .analysis import RBAnalysis, RBOptions
from .core import RBCfg, RBExp, RBResult
from .form import make_form

RBRun: TypeAlias = RunRecord[RBCfg, RBResult]


class TwoQubitRBAdapter(BaseAdapter[RBCfg, RBRun, RBAnalysis]):
    """Fixed-capture paired RB, saved-source analysis and canonical native IO.

    No calibration writeback or automatic acceptance is offered.
    """

    exp_cls = RBExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[RBCfg]:
        """Create editable pulse/correction inputs from one fixed Open/Reset capture."""
        return make_form(ctx)

    def run(self, req: RunRequest, cfg: RBCfg, *, context: RunContext) -> RBRun:
        """Keep cfg/raw paired; caller checks cancel_signal.raise_if_error()."""
        del req
        return RunRecord(cfg=cfg, result=RBExp().run(cfg, context=context))

    def analyze(self, req: AnalyzeRequest[RBRun, None], *, plots: Plots) -> RBAnalysis:
        """Count joint00 and fit only the saved source, never the current Form."""
        return RBExp().analyze(req.run_result, RBOptions(), plots=plots)
