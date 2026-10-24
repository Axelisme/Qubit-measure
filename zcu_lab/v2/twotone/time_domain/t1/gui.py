from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParameterWriteback,
    ParamMeta,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)

from zcu_lab.v2.support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.twotone.time_domain.t1.core import (
    T1Analysis,
    T1AnalyzeOptions,
    T1Cfg,
    T1Exp,
    T1Result,
)

from .form import make_form

if TYPE_CHECKING:
    from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

T1RunResult: TypeAlias = RunRecord[T1Cfg, T1Result]


@dataclass
class T1AnalyzeParams:
    dual_exp: Annotated[bool, ParamMeta(label="Dual exponential")] = False
    skip: Annotated[int, ParamMeta(label="Skip leading points")] = 0


class T1Summary(BaseModel):
    """Analysis summary in native units; null retains an unavailable estimate."""

    t1: float | None = Field(
        description="Owner t1 in us; null retains an unavailable estimate."
    )

    t1_err: float | None = Field(
        description="Owner t1_err in us; null retains an unavailable estimate."
    )

    t1b: float | None = Field(
        description="Owner t1b in us; null retains an unavailable estimate."
    )

    t1b_err: float | None = Field(
        description="Owner t1b_err in us; null retains an unavailable estimate."
    )


@dataclass(frozen=True)
class T1AnalyzeResult:
    """GUI summary over the unchanged, typed core analysis."""

    analysis: T1Analysis

    @property
    def t1(self) -> float:
        return self.analysis.t1

    @property
    def t1_err(self) -> float:
        return self.analysis.t1_err

    def summary(self) -> T1Summary:
        """Return the named scalar summary consumed by recipes, in native units."""
        return T1Summary(
            t1=self.analysis.t1,
            t1_err=self.analysis.t1_err,
            t1b=self.analysis.t1b,
            t1b_err=self.analysis.t1b_err,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            **self.summary().model_dump(),
            "fit_quality": fit_quality_summary(self.analysis.fit_quality),
        }


def _T1_make_form(ctx: FormContext) -> Form[T1Cfg]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _T1_analyze(
    req: AnalyzeRequest[T1RunResult, T1AnalyzeParams], *, plots: Plots
) -> T1AnalyzeResult:
    params = req.analyze_params
    if params.skip < 0:
        raise ValueError("Skip leading points must be nonnegative")
    analysis = T1Exp().analyze(
        req.run_result,
        T1AnalyzeOptions(dual_exp=params.dual_exp, skip=params.skip),
        plots=plots,
    )
    return T1AnalyzeResult(analysis)


def _T1_get_writeback_items(
    req: WritebackRequest[T1RunResult, T1AnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose fitted T1 parameters and a captured-component sample.

    req supplies the source result and fixed Run context. Sample fields use
    us and target the Run's resolved $qubit, never the live selection. Without
    a parameter capture, retain parameter proposals only so loaded data can
    still be analyzed; Apply rejects the missing capture. Unbound $qubit
    with a capture raises RoleResolutionError. WritebackService supplies
    bias coordinates or omits the sample with its existing public reason.
    No write or device access occurs here; this Run leaves devices fixed.
    """
    result = req.analyze_result
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.t1",
            description="T1 relaxation time (us)",
            proposed_value=result.t1,
            stderr=result.t1_err,
        ),
    ]
    parameters = req.ctx.parameters
    if parameters is not None:
        items.append(
            SampleWriteback(
                coordinate_device="$qubit.flux_source",
                target_name=parameters.resolve_path("$qubit"),
                description="T1 sample at the captured bias",
                proposed_value={
                    "t1_us": result.t1,
                    "t1_err_us": result.t1_err,
                },
            )
        )
    return items


def _run(cfg: T1Cfg, *, context: RunContext) -> RunRecord[T1Cfg, T1Result]:
    """Acquire a trace and retain the experiment configuration with its result."""
    return RunRecord(cfg=cfg, result=T1Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=T1Exp.AXES_SPEC,
    make_form=_T1_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_T1_analyze,
        options=lambda source, env: T1AnalyzeParams(),
        writeback=_T1_get_writeback_items,
    ),
)
