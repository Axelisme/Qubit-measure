"""Measure attachment for conditional Ramsey; acquisition never writes calibration."""

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

from ._analysis import IdleZZAnalysis
from .core import IdleZZCfg, IdleZZExp, IdleZZResult
from .form import make_form

IdleZZRun: TypeAlias = RunRecord[IdleZZCfg, IdleZZResult]


class IdleZZAdapter(BaseAdapter[IdleZZCfg, IdleZZRun, IdleZZAnalysis]):
    """Fixed-capture Form, thermal Ramsey acquisition and native restore."""

    exp_cls = IdleZZExp
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, load_data=True
    )
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[IdleZZCfg]:
        """Resolve roles/full tone mapping and calibrated pulse seeds at Open/Reset."""
        return make_form(ctx)

    def run(self, req: RunRequest, cfg: IdleZZCfg, *, context: RunContext) -> IdleZZRun:
        """Keep original cfg with acquired raw, including cancelled partial rounds.

        Context owns external acquisition, optional named device, plots and stop.
        The operation owner checks cancel_signal.raise_if_error() after return;
        invalid acquired IQ cannot masquerade as a successful cancelled run.
        """
        del req
        return RunRecord(cfg=cfg, result=IdleZZExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[IdleZZRun, None],
        *,
        plots: Plots,
    ) -> IdleZZAnalysis:
        """Fit saved cfg/raw only; current aliases, Form and library are ignored."""
        return IdleZZExp().analyze(req.run_result, None, plots=plots)
