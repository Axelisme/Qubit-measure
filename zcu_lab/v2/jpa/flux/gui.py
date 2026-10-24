"""jpa/flux GUI adapter — JPA flux-device calibration sweep.

Owns the jpa/flux cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

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

from zcu_lab.v2.jpa.flux.core import FluxCfg, FluxExp, FluxResult
from zcu_lab.v2.jpa.flux.form import make_form

JpaFluxRunResult: TypeAlias = RunRecord[FluxCfg, FluxResult]


@dataclass
class JpaFluxAnalyzeResult(AnalyzeResultBase):
    best_flux: float


def _relabel_flux_figure(fig: Figure, best_flux: float) -> None:
    """Neutral JPA flux device value wording on the GUI review figure.

    The core analysis figure still labels its flux axis and the optimum legend
    with 'a.u.'; the GUI review figure must use the neutral 'JPA flux device
    value' vocabulary (A6) without claiming a physical-unit migration. The
    relabel happens at the adapter analysis boundary, in place, so the review
    figure wording is adapter-owned while the core figure stays untouched.
    """
    for ax in fig.axes:
        ax.set_xlabel("JPA flux device value")
        for line in ax.get_lines():
            label = line.get_label()
            if isinstance(label, str) and label.startswith("best JPA flux"):
                line.set_label(f"best JPA flux device value = {best_flux:.2g}")
        legend = ax.get_legend()
        if legend is not None:
            ax.legend()


class JpaFluxAdapter(BaseAdapter[FluxCfg, JpaFluxRunResult, JpaFluxAnalyzeResult]):
    exp_cls = FluxExp

    @staticmethod
    def make_form(ctx: FormContext) -> Form[FluxCfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: FluxCfg,
        *,
        context: RunContext,
    ) -> JpaFluxRunResult:
        del req
        return RunRecord(cfg, FluxExp().run(cfg, context=context))

    def analyze(
        self, req: AnalyzeRequest[JpaFluxRunResult, None], *, plots: Plots
    ) -> JpaFluxAnalyzeResult:
        answer = FluxExp().analyze(req.run_result, None, plots=plots)
        _relabel_flux_figure(plots["fit"], answer.best_flux)
        return JpaFluxAnalyzeResult(best_flux=answer.best_flux)

    def get_writeback_items(
        self, req: WritebackRequest[JpaFluxRunResult, JpaFluxAnalyzeResult]
    ) -> Sequence[WritebackItem]:
        return [
            ParameterWriteback(
                target_name="$jpa.flux",
                description="Best JPA flux device value",
                proposed_value=req.analyze_result.best_flux,
            )
        ]
