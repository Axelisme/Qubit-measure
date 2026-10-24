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

from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import (
    FreqGainAnalyzeOptions,
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_lab.v2.twotone.ro_optimize.freq_gain.form import make_form

RoOptFreqGainRunResult: TypeAlias = RunRecord[FreqGainCfg, FreqGainResult]


@dataclass
class RoOptFreqGainAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class RoOptFreqGainAnalyzeResult(AnalyzeResultBase):
    best_freq: float
    best_gain: float


def _RoOptFreqGain_make_form(ctx: FormContext) -> Form[FreqGainCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _RoOptFreqGain_analyze(
    req: AnalyzeRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeParams],
    *,
    plots: Plots,
) -> RoOptFreqGainAnalyzeResult:
    params = req.analyze_params
    options = FreqGainAnalyzeOptions(
        smooth=params.smooth,
        smooth_method=params.smooth_method,
    )
    result = FreqGainExp().analyze(req.run_result, options, plots=plots)
    return RoOptFreqGainAnalyzeResult(
        best_freq=result.best_freq,
        best_gain=result.best_gain,
    )


def _RoOptFreqGain_get_writeback_items(
    req: WritebackRequest[RoOptFreqGainRunResult, RoOptFreqGainAnalyzeResult],
) -> Sequence[WritebackItem]:
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.ext.best_ro_freq",
            description="Optimal readout frequency (MHz)",
            proposed_value=result.best_freq,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.best_ro_gain",
            description="Optimal readout gain (a.u.)",
            proposed_value=result.best_gain,
        ),
    ]
    items.extend(
        readout_dpm_writeback_items(
            req.ctx,
            req.run_result.cfg,
            proposed={
                "best_ro_freq": result.best_freq,
                "best_ro_gain": result.best_gain,
            },
        )
    )
    return items


def _run(
    cfg: FreqGainCfg, *, context: RunContext
) -> RunRecord[FreqGainCfg, FreqGainResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=FreqGainExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FreqGainExp.AXES_SPEC,
    make_form=_RoOptFreqGain_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_RoOptFreqGain_analyze,
        options=lambda source, env: RoOptFreqGainAnalyzeParams(),
        writeback=_RoOptFreqGain_get_writeback_items,
    ),
)
