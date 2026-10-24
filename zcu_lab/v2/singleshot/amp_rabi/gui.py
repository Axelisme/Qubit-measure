from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_rabi import rabi_calibration_writeback
from zcu_lab.v2.singleshot.amp_rabi.core import (
    AmpRabiAnalyzeOptions,
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiFit,
    AmpRabiResult,
)
from zcu_lab.v2.singleshot.amp_rabi.form import make_form

SsAmpRabiRunResult = RunRecord[AmpRabiCfg, AmpRabiResult]


@dataclass
class SsAmpRabiAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"


@dataclass
class SsAmpRabiAnalyzeResult(AnalyzeResultBase):
    pi_gain: float
    pi_gain_error: float
    pi2_gain: float
    pi2_gain_error: float
    frequency: float
    amplitude: float
    fit_result: AmpRabiFit


def _analyze(
    req: AnalyzeRequest[SsAmpRabiRunResult, SsAmpRabiAnalyzeParams],
    *,
    plots: Plots,
) -> SsAmpRabiAnalyzeResult:
    fit = AmpRabiExp().analyze(
        req.run_result,
        AmpRabiAnalyzeOptions(initial_state=req.analyze_params.initial_state),
        plots=plots,
    )
    return SsAmpRabiAnalyzeResult(
        fit.pi_gain,
        fit.pi_gain_error,
        fit.pi2_gain,
        fit.pi2_gain_error,
        fit.frequency,
        fit.amplitude,
        fit,
    )


def _writeback(
    req: WritebackRequest[SsAmpRabiRunResult, SsAmpRabiAnalyzeResult],
) -> Sequence[WritebackItem]:
    return rabi_calibration_writeback(req.analyze_result.fit_result.joint_fit)


def _run(
    cfg: AmpRabiCfg, *, context: RunContext
) -> RunRecord[AmpRabiCfg, AmpRabiResult]:
    return RunRecord(cfg=cfg, result=AmpRabiExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=AmpRabiExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: SsAmpRabiAnalyzeParams(),
        writeback=_writeback,
    ),
)
