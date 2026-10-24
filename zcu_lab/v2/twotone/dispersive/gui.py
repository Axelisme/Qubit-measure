"""Measure GUI attachment for dispersive-shift spectroscopy."""

from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    RunRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.twotone.dispersive.core import (
    DispersiveAnalysis,
    DispersiveAnalyzeOptions,
    DispersiveCfg,
    DispersiveExp,
    DispersiveResult,
)
from zcu_lab.v2.twotone.dispersive.form import make_form

DispersiveRunResult: TypeAlias = RunRecord[DispersiveCfg, DispersiveResult]


class DispersiveAdapter(
    BaseAdapter[
        DispersiveCfg,
        DispersiveRunResult,
        DispersiveAnalysis,
        DispersiveAnalyzeOptions,
    ]
):
    """Acquire paired readout spectra and fit their dispersive shift."""

    exp_cls = DispersiveExp
    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @staticmethod
    def make_form(ctx: FormContext) -> Form[DispersiveCfg]:
        """Build fresh inputs from the fixed Open/Reset capture."""
        return make_form(ctx)

    def run(
        self,
        req: RunRequest,
        cfg: DispersiveCfg,
        *,
        context: RunContext,
    ) -> DispersiveRunResult:
        """Sweep readout with preparation off/on and retain this run's cfg."""
        del req
        return RunRecord(cfg=cfg, result=DispersiveExp().run(cfg, context=context))

    def analyze(
        self,
        req: AnalyzeRequest[DispersiveRunResult, DispersiveAnalyzeOptions],
        *,
        plots: Plots,
    ) -> DispersiveAnalysis:
        """Fit the source spectra, publish the fit figure, and return MHz values."""
        return DispersiveExp().analyze(
            req.run_result,
            req.analyze_params,
            plots=plots,
        )
