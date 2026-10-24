"""jpa/auto_optimize GUI adapter — joint JPA flux/frequency/power optimization.

Owns the jpa/auto_optimize cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from matplotlib.figure import Figure
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    RunRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.auto_optimize.core import (
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeResult,
)
from zcu_lab.v2.jpa.auto_optimize.form import make_form

JpaAutoRunResult = RunRecord[JPAOptCfg, JPAOptimizeResult]


def _relabel_auto_figure(fig: Figure) -> None:
    """Neutral 'JPA flux device value' wording on the GUI review figure.

    The core auto-optimize analysis figure labels its flux subplot 'JPA Flux
    value (a.u.)'; the GUI review figure must use the neutral vocabulary of
    the selected flux device without claiming a physical-unit migration. Only
    the flux axis is relabelled — the frequency and power axes keep their own
    quantities.
    """
    for ax in fig.axes:
        if "Flux" in ax.get_xlabel():
            ax.set_xlabel("JPA flux device value")


@dataclass
class JpaAutoAnalyzeResult(AnalyzeResultBase):
    best_flux: float
    best_freq: float
    best_power: float


class JpaAutoOptimizeAdapter(
    BaseAdapter[JPAOptCfg, JpaAutoRunResult, JpaAutoAnalyzeResult]
):
    exp_cls = AutoOptimizeExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[JPAOptCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: JPAOptCfg,
        *,
        context: RunContext,
    ) -> JpaAutoRunResult:
        del req
        return RunRecord(cfg, AutoOptimizeExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaAutoRunResult, None], *, plots: Plots
    ) -> JpaAutoAnalyzeResult:
        analysis = AutoOptimizeExp().analyze(req.run_result, None, plots=plots)
        _relabel_auto_figure(plots["fit"])
        return JpaAutoAnalyzeResult(
            best_flux=analysis.best_flux,
            best_freq=analysis.best_freq,
            best_power=analysis.best_power,
        )

    def get_writeback_items(
        self, req: WritebackRequest[JpaAutoRunResult, JpaAutoAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        result = req.analyze_result
        return [
            ParameterWriteback(
                target_name="$jpa.flux",
                description="Best JPA flux device value",
                proposed_value=result.best_flux,
            ),
            ParameterWriteback(
                target_name="$jpa.pump_freq",
                description="Best JPA pump frequency (MHz)",
                proposed_value=result.best_freq,
            ),
            ParameterWriteback(
                target_name="$jpa.pump_power",
                description="Best JPA pump power (dBm)",
                proposed_value=result.best_power,
            ),
        ]
