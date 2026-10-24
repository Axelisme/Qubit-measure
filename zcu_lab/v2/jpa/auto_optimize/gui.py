"""jpa/auto_optimize GUI adapter — joint JPA flux/frequency/power optimization.

Owns the jpa/auto_optimize cfg definition, run/analyze/writeback policy and the
operator guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.jpa.auto_optimize.core import (
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeAnalysis,
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


def _analyze(
    req: AnalyzeRequest[JpaAutoRunResult, None], *, plots: Plots
) -> JPAOptimizeAnalysis:
    analysis = AutoOptimizeExp().analyze(req.run_result, None, plots=plots)
    _relabel_auto_figure(plots["fit"])
    return analysis


def _get_writeback_items(
    req: WritebackRequest[JpaAutoRunResult, JPAOptimizeAnalysis],
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


def run(
    cfg: JPAOptCfg, *, context: RunContext
) -> RunRecord[JPAOptCfg, JPAOptimizeResult]:
    """Acquire with cfg and retain it as the analysis and save source."""
    return RunRecord(cfg=cfg, result=AutoOptimizeExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=AutoOptimizeExp.AXES_SPEC,
    make_form=make_form,
    run=run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
