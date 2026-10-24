"""jpa/freq GUI adapter — JPA pump-frequency tracer.

Owns the jpa/freq cfg definition, run/analyze/writeback policy and the operator
guide. The core experiment lives in ``zcu_lab.v2.jpa``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

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

from zcu_lab.v2.jpa.freq.core import FreqCfg, FreqExp, FreqResult
from zcu_lab.v2.jpa.freq.form import make_form

JpaFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class JpaFreqAnalyzeResult(AnalyzeResultBase):
    best_freq: float


def _analyze(
    req: AnalyzeRequest[JpaFreqRunResult, None], *, plots: Plots
) -> JpaFreqAnalyzeResult:
    answer = FreqExp().analyze(req.run_result, None, plots=plots)
    return JpaFreqAnalyzeResult(best_freq=answer.best_freq)


def _get_writeback_items(
    req: WritebackRequest[JpaFreqRunResult, JpaFreqAnalyzeResult],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback(
            target_name="$jpa.pump_freq",
            description="Best JPA pump frequency (MHz)",
            proposed_value=req.analyze_result.best_freq,
        )
    ]


DEFINITION = Adapter(
    spec=FreqExp.AXES_SPEC,
    make_form=make_form,
    run=FreqExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_get_writeback_items),
)
