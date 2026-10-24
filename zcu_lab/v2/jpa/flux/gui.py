"""jpa/flux GUI adapter — JPA flux-device calibration sweep.

Owns the jpa/flux cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from matplotlib.figure import Figure
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
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


def _analyze(
    req: AnalyzeRequest[JpaFluxRunResult, None], *, plots: Plots
) -> JpaFluxAnalyzeResult:
    answer = FluxExp().analyze(req.run_result, None, plots=plots)
    _relabel_flux_figure(plots["fit"], answer.best_flux)
    return JpaFluxAnalyzeResult(best_flux=answer.best_flux)


def _get_writeback_items(
    req: WritebackRequest[JpaFluxRunResult, JpaFluxAnalyzeResult],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback(
            target_name="$jpa.flux",
            description="Best JPA flux device value",
            proposed_value=req.analyze_result.best_flux,
        )
    ]


def run(cfg: FluxCfg, *, context: RunContext) -> RunRecord[FluxCfg, FluxResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=FluxExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FluxExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
