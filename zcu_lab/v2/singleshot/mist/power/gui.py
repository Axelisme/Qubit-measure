from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParamMeta,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_shared import (
    read_analysis_float,
    read_confusion_matrix,
)
from zcu_lab.v2.singleshot.mist.power.core import (
    PowerAnalyzeOptions,
    PowerCfg,
    PowerExp,
    PowerResult,
)
from zcu_lab.v2.singleshot.mist.power.form import make_form

MistPowerRunResult: TypeAlias = RunRecord[PowerCfg, PowerResult]


@dataclass
class MistPowerAnalyzeParams:
    """MIST plot settings; log_scale uses a logarithmic gain/photon-number axis."""

    log_scale: Annotated[bool, ParamMeta(label="Logarithmic x-axis")] = False


def _analyze(
    req: AnalyzeRequest[MistPowerRunResult, MistPowerAnalyzeParams],
    *,
    plots: Plots,
) -> None:
    options = PowerAnalyzeOptions(
        ac_coeff=read_analysis_float(req.parameters, "$qubit.ext.ac_stark_coeff"),
        log_scale=req.analyze_params.log_scale,
        confusion_matrix=read_confusion_matrix(req.parameters),
    )
    PowerExp().analyze(req.run_result, options, plots=plots)
    return


DEFINITION = Adapter(
    spec=PowerExp.AXES_SPEC,
    make_form=make_form,
    run=PowerExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze, options=lambda source, env: MistPowerAnalyzeParams()
    ),
)
