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

from zcu_lab.v2.singleshot.mist.power_freq.core import (
    FreqPowerAnalyzeOptions,
    FreqPowerCfg,
    FreqPowerExp,
    FreqPowerResult,
)
from zcu_lab.v2.singleshot.mist.power_freq.form import make_form
from zcu_lab.v2.support.measure.singleshot_shared import (
    read_confusion_matrix,
)

MistPowerFreqRunResult: TypeAlias = RunRecord[FreqPowerCfg, FreqPowerResult]


def _analyze(
    req: AnalyzeRequest[MistPowerFreqRunResult, None],
    *,
    plots: Plots,
) -> None:
    options = FreqPowerAnalyzeOptions(
        confusion_matrix=read_confusion_matrix(req.parameters),
    )
    FreqPowerExp().analyze(req.run_result, options, plots=plots)
    return


def _run(
    cfg: FreqPowerCfg, *, context: RunContext
) -> RunRecord[FreqPowerCfg, FreqPowerResult]:
    return RunRecord(cfg=cfg, result=FreqPowerExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FreqPowerExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(fn=_analyze),
)
