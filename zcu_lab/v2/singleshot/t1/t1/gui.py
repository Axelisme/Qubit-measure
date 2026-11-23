from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.analysis.fitting.multi_decay import ExcitedPopulationDecayTimes
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.singleshot.t1.t1.core import T1AnalyzeOptions, T1Cfg, T1Exp, T1Result
from zcu_lab.v2.singleshot.t1.t1.form import make_form

SsT1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


def _analyze(
    req: AnalyzeRequest[SsT1RunResult, None], *, plots: Plots
) -> ExcitedPopulationDecayTimes:
    return T1Exp().analyze(req.run_result, T1AnalyzeOptions(), plots=plots)


def _writeback(
    req: WritebackRequest[SsT1RunResult, ExcitedPopulationDecayTimes],
) -> list[ParameterWriteback]:
    return [
        ParameterWriteback(
            target_name="$qubit.t1",
            description="T1 of the dominant e-population decay mode (µs).",
            proposed_value=req.analyze_result.t1,
        ),
    ]


def _summary_lines(source: SsT1RunResult) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps×Rounds: {cfg.reps}×{cfg.rounds}",)


def _run(cfg: T1Cfg, *, context: RunContext) -> RunRecord[T1Cfg, T1Result]:
    return RunRecord(cfg=cfg, result=T1Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=T1Exp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_writeback, summary_lines=_summary_lines),
)
