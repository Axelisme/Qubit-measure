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
    ParameterWriteback,
    ParamMeta,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.writeback_helpers import readout_dpm_writeback_items
from zcu_lab.v2.twotone.ro_optimize.freq.core import (
    FreqAnalysis,
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.ro_optimize.freq.form import make_form

RoOptFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class RoOptFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 2.0


def _RoOptFreq_analyze(
    req: AnalyzeRequest[RoOptFreqRunResult, RoOptFreqAnalyzeParams], *, plots: Plots
) -> FreqAnalysis:
    params = req.analyze_params
    options = FreqAnalyzeOptions(
        smooth=params.smooth,
        smooth_method=params.smooth_method,
    )
    return FreqExp().analyze(req.run_result, options, plots=plots)


def _RoOptFreq_get_writeback_items(
    req: WritebackRequest[RoOptFreqRunResult, FreqAnalysis],
) -> Sequence[WritebackItem]:
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.ext.best_ro_freq",
            description="Optimal readout frequency (MHz)",
            proposed_value=result.best_freq,
        ),
    ]
    items.extend(
        readout_dpm_writeback_items(
            req.ctx,
            req.run_result.cfg,
            proposed={"best_ro_freq": result.best_freq},
        )
    )
    return items


def _RoOptFreq_make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _run(cfg: FreqCfg, *, context: RunContext) -> RunRecord[FreqCfg, FreqResult]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=FreqExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=FreqExp.AXES_SPEC,
    make_form=_RoOptFreq_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_RoOptFreq_analyze,
        options=lambda source, env: RoOptFreqAnalyzeParams(),
        writeback=_RoOptFreq_get_writeback_items,
    ),
)
