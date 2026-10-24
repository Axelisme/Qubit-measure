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

from zcu_lab.v2.twotone.reset.dual_tone.freq.core import (
    FreqAnalyzeOptions,
    FreqCfg,
    FreqExp,
    FreqResult,
)
from zcu_lab.v2.twotone.reset.dual_tone.freq.form import make_form

DualToneFreqRunResult: TypeAlias = RunRecord[FreqCfg, FreqResult]


@dataclass
class DualToneFreqAnalyzeParams:
    smooth_method: Annotated[
        Literal["wavelet", "gaussian"], ParamMeta(label="Smooth method")
    ] = "wavelet"
    smooth: Annotated[float, ParamMeta(label="Smooth strength", decimals=2)] = 1.0


@dataclass
class DualToneFreqAnalyzeResult(AnalyzeResultBase):
    freq1: float
    freq2: float


def _DualToneFreq_make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _DualToneFreq_analyze(
    req: AnalyzeRequest[DualToneFreqRunResult, DualToneFreqAnalyzeParams],
    *,
    plots: Plots,
) -> DualToneFreqAnalyzeResult:
    params = req.analyze_params
    result = FreqExp().analyze(
        req.run_result,
        FreqAnalyzeOptions(smooth=params.smooth, smooth_method=params.smooth_method),
        plots=plots,
    )
    return DualToneFreqAnalyzeResult(freq1=result.freq1, freq2=result.freq2)


def _DualToneFreq_get_writeback_items(
    req: WritebackRequest[DualToneFreqRunResult, DualToneFreqAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose this analysis's two frequencies to captured qubit ext."""
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.ext.reset_f1",
            description="Dual-tone reset frequency 1 (MHz)",
            proposed_value=result.freq1,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.reset_f2",
            description="Dual-tone reset frequency 2 (MHz)",
            proposed_value=result.freq2,
        ),
    ]
    return items


DEFINITION = Adapter(
    spec=FreqExp.AXES_SPEC,
    make_form=_DualToneFreq_make_form,
    run=FreqExp().run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_DualToneFreq_analyze,
        options=lambda source, env: DualToneFreqAnalyzeParams(),
        writeback=_DualToneFreq_get_writeback_items,
    ),
)
