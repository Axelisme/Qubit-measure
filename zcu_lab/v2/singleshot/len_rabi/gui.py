from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from zcu_tools.analysis.fitting.singleshot_rabi import RabiJointFitResult
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.singleshot_rabi import rabi_calibration_writeback
from zcu_lab.v2.singleshot.len_rabi.core import (
    LenRabiAnalyzeOptions,
    LenRabiCfg,
    LenRabiExp,
    LenRabiResult,
)
from zcu_lab.v2.singleshot.len_rabi.form import make_form

# ``LenRabiExp`` from ``singleshot`` — sweeps the qubit-drive pulse *length* and
# preserves every raw IQ shot. Analysis derives populations from that canonical
# raw result rather than persisting a second population representation.
SsLenRabiRunResult: TypeAlias = RunRecord[LenRabiCfg, LenRabiResult]


@dataclass
class SsLenRabiAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"
    decay: Annotated[bool, ParamMeta(label="Fit decay envelope")] = True
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset")] = False


def _analyze(
    req: AnalyzeRequest[SsLenRabiRunResult, SsLenRabiAnalyzeParams], *, plots: Plots
) -> RabiJointFitResult:
    return LenRabiExp().analyze(
        req.run_result,
        LenRabiAnalyzeOptions(
            decay=req.analyze_params.decay,
            fit_phase=req.analyze_params.fit_phase,
            initial_state=req.analyze_params.initial_state,
        ),
        plots=plots,
    )


def _writeback(
    req: WritebackRequest[SsLenRabiRunResult, RabiJointFitResult],
) -> Sequence[WritebackItem]:
    return rabi_calibration_writeback(req.analyze_result)


def _run(
    cfg: LenRabiCfg, *, context: RunContext
) -> RunRecord[LenRabiCfg, LenRabiResult]:
    return RunRecord(cfg=cfg, result=LenRabiExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=LenRabiExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: SsLenRabiAnalyzeParams(),
        writeback=_writeback,
    ),
)
