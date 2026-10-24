from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

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

from zcu_lab.v2._support.measure.singleshot_shared import read_confusion_matrix
from zcu_lab.v2.singleshot.t1.t1_with_tone.core import (
    T1WithToneAnalysis,
    T1WithToneAnalyzeOptions,
    T1WithToneCfg,
    T1WithToneExp,
    T1WithToneResult,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone.form import make_form

# The numeric t1 is written to the t1_with_tone parameter context key.
SsT1ToneRunResult: TypeAlias = RunRecord[T1WithToneCfg, T1WithToneResult]


def _analyze(
    req: AnalyzeRequest[SsT1ToneRunResult, None], *, plots: Plots
) -> T1WithToneAnalysis:
    confusion = read_confusion_matrix(req.parameters)
    return T1WithToneExp().analyze(
        req.run_result,
        T1WithToneAnalyzeOptions(confusion_matrix=confusion),
        plots=plots,
    )


def _writeback(
    req: WritebackRequest[SsT1ToneRunResult, T1WithToneAnalysis],
) -> Sequence[WritebackItem]:
    return [
        ParameterWriteback(
            target_name="$qubit.ext.t1_with_tone",
            description="T1 with probe tone (us)",
            proposed_value=req.analyze_result.t1,
        ),
    ]


def _run(
    cfg: T1WithToneCfg, *, context: RunContext
) -> RunRecord[T1WithToneCfg, T1WithToneResult]:
    return RunRecord(cfg=cfg, result=T1WithToneExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=T1WithToneExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze, writeback=_writeback),
)
