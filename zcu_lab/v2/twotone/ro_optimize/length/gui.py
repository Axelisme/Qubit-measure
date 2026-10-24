from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.length.core import (
    LengthAnalyzeOptions,
    LengthCfg,
    LengthExp,
    LengthResult,
)
from zcu_lab.v2.twotone.ro_optimize.length.form import make_form

RoOptLengthRunResult: TypeAlias = RunRecord[LengthCfg, LengthResult]


@dataclass
class RoOptLengthAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0
    # GUI-facing name for LengthAnalyzeOptions.t0: this is a duration
    # normalization overhead, not a penalty strength. Small positive values
    # produce stronger short-readout bias than large positive values.
    duration_t0: Annotated[float | None, ParamMeta(label="Duration t0 (us)")] = None


@dataclass
class RoOptLengthAnalyzeResult(AnalyzeResultBase):
    best_length: float


def _RoOptLength_make_form(ctx: FormContext) -> Form[LengthCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _RoOptLength_analyze(
    req: AnalyzeRequest[RoOptLengthRunResult, RoOptLengthAnalyzeParams],
    *,
    plots: Plots,
) -> RoOptLengthAnalyzeResult:
    params = req.analyze_params
    options = LengthAnalyzeOptions(
        t0=params.duration_t0,
        smooth=params.smooth,
        smooth_method=params.smooth_method,
    )
    result = LengthExp().analyze(req.run_result, options, plots=plots)
    return RoOptLengthAnalyzeResult(
        best_length=result.best_length,
    )


def _RoOptLength_get_writeback_items(
    req: WritebackRequest[RoOptLengthRunResult, RoOptLengthAnalyzeResult],
) -> Sequence[WritebackItem]:
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.ext.best_ro_length",
            description="Optimal readout length (us)",
            proposed_value=result.best_length,
        ),
    ]
    items.extend(
        readout_dpm_writeback_items(
            req.ctx,
            req.run_result.cfg,
            proposed={"best_ro_length": result.best_length},
        )
    )
    return items


def _run(cfg: LengthCfg, *, context: RunContext) -> RunRecord[LengthCfg, LengthResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=LengthExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=LengthExp.AXES_SPEC,
    make_form=_RoOptLength_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_RoOptLength_analyze,
        options=lambda source, env: RoOptLengthAnalyzeParams(),
        writeback=_RoOptLength_get_writeback_items,
    ),
)
