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

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.mist.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqDepExp,
    FreqResult,
)
from zcu_lab.v2.singleshot.mist.freq.form import make_form

MistFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


def _analyze(
    req: AnalyzeRequest[MistFreqRunResult, None],
    *,
    plots: Plots,
) -> None:
    options = FreqAnalyzeOptions(
        confusion_matrix=read_confusion_matrix(req.parameters),
    )
    FreqDepExp().analyze(req.run_result, options, plots=plots)
    return


def _run(cfg: FreqCfg, *, context: RunContext) -> RunRecord[FreqCfg, FreqResult]:
    return RunRecord(cfg=cfg, result=FreqDepExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FreqDepExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
