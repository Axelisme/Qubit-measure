"""Measure attachment for thermal four-state joint readout."""

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from .core import (
    JointReadoutAnalysis,
    JointReadoutCfg,
    JointReadoutExp,
    JointReadoutResult,
)
from .form import make_form

JointReadoutRun: TypeAlias = RunRecord[JointReadoutCfg, JointReadoutResult]


class JointReadoutAdapter(
    BaseAdapter[JointReadoutCfg, JointReadoutRun, JointReadoutAnalysis]
):
    """Open, acquire, load and classify joint raw shots without calibration writes."""

    exp_cls = JointReadoutExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[JointReadoutCfg]:
        """Use one fixed capture for roles, full tones, pulses and GE centers."""
        return make_form(ctx)

    def run(
        self, req: RunRequest, cfg: JointReadoutCfg, *, context: RunContext
    ) -> JointReadoutRun:
        """Pair the unchanged full cfg with raw IQ, including cancelled partial data.

        Context owns hardware, plots and the shared stop signal. The caller must
        check raise_if_error to distinguish acquisition errors from cancellation.
        """
        del req
        return RunRecord(cfg=cfg, result=JointReadoutExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JointReadoutRun, None], *, plots: Plots
    ) -> JointReadoutAnalysis:
        """Classify complete branches using saved cfg; never current parameters."""
        return JointReadoutExp().analyze(req.run_result, None, plots=plots)

    def get_writeback_items(
        self, req: WritebackRequest[JointReadoutRun, JointReadoutAnalysis]
    ) -> Sequence[WritebackItem]:
        """Return no proposals: this thermal matrix is a diagnostic, not a correction."""
        del req
        return []
