from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

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

from zcu_lab.v2.twotone.reset.bath.freq.core import (
    FreqGainAnalyzeOptions,
    FreqGainCfg,
    FreqGainExp,
    FreqGainResult,
)
from zcu_lab.v2.twotone.reset.bath.freq.form import make_form

BathFreqGainRunResult: TypeAlias = RunRecord[FreqGainCfg, FreqGainResult]


@dataclass
class BathFreqGainAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class BathFreqGainAnalyzeResult(AnalyzeResultBase):
    gain: float
    freq: float


def _BathFreqGain_make_form(ctx: FormContext) -> Form[FreqGainCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _BathFreqGain_analyze(
    req: AnalyzeRequest[BathFreqGainRunResult, BathFreqGainAnalyzeParams],
    *,
    plots: Plots,
) -> BathFreqGainAnalyzeResult:
    params = req.analyze_params
    analysis = FreqGainExp().analyze(
        req.run_result,
        FreqGainAnalyzeOptions(
            smooth=params.smooth, smooth_method=params.smooth_method
        ),
        plots=plots,
    )
    return BathFreqGainAnalyzeResult(gain=analysis.gain, freq=analysis.freq)


def _BathFreqGain_get_writeback_items(
    req: WritebackRequest[BathFreqGainRunResult, BathFreqGainAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose this analysis's cavity gain then frequency to qubit ext."""
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.ext.reset_bath_gain",
            description="Bath-reset cavity gain (a.u.)",
            proposed_value=result.gain,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.reset_bath_freq",
            description="Bath-reset cavity frequency (MHz)",
            proposed_value=result.freq,
        ),
    ]
    return items


DEFINITION = Adapter(
    spec=FreqGainExp.AXES_SPEC,
    make_form=_BathFreqGain_make_form,
    run=FreqGainExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_BathFreqGain_analyze,
        options=lambda source, env: BathFreqGainAnalyzeParams(),
        writeback=_BathFreqGain_get_writeback_items,
    ),
)
