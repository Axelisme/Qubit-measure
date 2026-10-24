"""jpa/power GUI adapter — JPA pump-power calibration sweep.

Owns the jpa/power cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from matplotlib.figure import Figure
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

from zcu_lab.v2.jpa.power.core import PowerCfg, PowerExp, PowerResult
from zcu_lab.v2.jpa.power.form import make_form

JpaPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class JpaPowerAnalyzeResult(AnalyzeResultBase):
    best_power: float


def _relabel_power_figure(fig: Figure) -> None:
    """Correct the GUI review figure's x-axis to the swept quantity.

    The core analysis figure mislabels its x-axis as 'JPA Frequency (MHz)'
    although the swept data is pump power in dBm; the GUI review figure must
    say 'JPA pump power (dBm)'. The relabel happens at the adapter analysis
    boundary, in place, leaving the core figure and the Result/writeback
    values untouched.
    """
    for ax in fig.axes:
        ax.set_xlabel("JPA pump power (dBm)")


def _analyze(
    req: AnalyzeRequest[JpaPowerRunResult, None], *, plots: Plots
) -> JpaPowerAnalyzeResult:
    answer = PowerExp().analyze(req.run_result, None, plots=plots)
    _relabel_power_figure(plots["fit"])
    return JpaPowerAnalyzeResult(best_power=answer.best_power)


def _get_writeback_items(
    req: WritebackRequest[JpaPowerRunResult, JpaPowerAnalyzeResult],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback(
            target_name="$jpa.pump_power",
            description="Best JPA pump power (dBm)",
            proposed_value=req.analyze_result.best_power,
        )
    ]


DEFINITION = Adapter(
    spec=PowerExp.AXES_SPEC,
    make_form=make_form,
    run=PowerExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
