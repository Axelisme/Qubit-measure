from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field
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
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.lookback.core import (
    LookbackAnalyzeOptions,
    LookbackCfg,
    LookbackExp,
    LookbackResult,
)
from zcu_lab.v2.lookback.form import make_form

LookbackRunResult: TypeAlias = RunRecord[LookbackCfg, LookbackResult]


@dataclass
class LookbackAnalyzeParams:
    ratio: Annotated[float, ParamMeta(label="Threshold ratio", decimals=3)] = 0.1
    smooth: Annotated[float, ParamMeta(label="Smooth sigma", decimals=3)] = 1.0
    plot_fit: Annotated[bool, ParamMeta(label="Plot fit")] = True


class LookbackSummary(BaseModel):
    """Analysis summary in native units; null retains an unavailable estimate."""

    predict_offset: float | None = Field(
        description="Owner predict_offset in us; null retains an unavailable estimate."
    )


@dataclass
class LookbackAnalyzeResult(AnalyzeResultBase):
    predict_offset: float

    def summary(self) -> LookbackSummary:
        """Return the named scalar summary consumed by recipes, in native units."""
        return LookbackSummary(
            predict_offset=self.predict_offset,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {**super().to_summary_dict(), **self.summary().model_dump()}


def _analyze(
    req: AnalyzeRequest[LookbackRunResult, LookbackAnalyzeParams],
    *,
    plots: Plots,
) -> LookbackAnalyzeResult:
    params = req.analyze_params
    answer = LookbackExp().analyze(
        req.run_result,
        LookbackAnalyzeOptions(
            ratio=params.ratio,
            smooth=params.smooth,
            plot_fit=params.plot_fit,
        ),
        plots=plots,
    )
    return LookbackAnalyzeResult(predict_offset=answer.predict_offset)


def _writeback(
    req: WritebackRequest[LookbackRunResult, LookbackAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose detected arrival time in us at captured resonator.time_of_fly.

    req supplies the analysis and its capture; source.cfg may be None.
    WritebackService resolves the destination on req.ctx.parameters and
    rejects missing capture or unbound roles. No live focus lookup or write
    occurs here.
    """
    return [
        ParameterWriteback(
            target_name="$resonator.time_of_fly",
            description="Readout trigger offset prediction (us)",
            proposed_value=req.analyze_result.predict_offset,
        )
    ]


def _run(
    cfg: LookbackCfg, *, context: RunContext
) -> RunRecord[LookbackCfg, LookbackResult]:
    return RunRecord(cfg=cfg, result=LookbackExp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=LookbackExp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: LookbackAnalyzeParams(),
        writeback=_writeback,
    ),
)
