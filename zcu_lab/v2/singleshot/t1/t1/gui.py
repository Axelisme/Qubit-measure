from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.singleshot.t1.t1.core import T1AnalyzeOptions, T1Cfg, T1Exp, T1Result
from zcu_lab.v2.singleshot.t1.t1.form import make_form
from zcu_lab.v2.support.measure.singleshot_shared import read_confusion_matrix

# Transition-rate analysis publishes figures without numeric writeback.
SsT1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


def _analyze(req: AnalyzeRequest[SsT1RunResult, None], *, plots: Plots) -> None:
    # ``confusion_matrix`` is the GE 3×3 readout-correction matrix from md.
    # ``skip`` is not exposed as a user knob — users can re-run with a shorter
    # sweep instead.
    confusion = read_confusion_matrix(req.parameters)
    T1Exp().analyze(
        req.run_result, T1AnalyzeOptions(confusion_matrix=confusion), plots=plots
    )
    return


def _run(cfg: T1Cfg, *, context: RunContext) -> RunRecord[T1Cfg, T1Result]:
    return RunRecord(cfg=cfg, result=T1Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=T1Exp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
