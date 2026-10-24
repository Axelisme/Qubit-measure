from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

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
    PostAnalysis,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    PostWritebackRequest,
    WritebackItem,
    WritebackRequest,
)

from zcu_lab.v2.singleshot.ge.core import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
)
from zcu_lab.v2.singleshot.ge.form import make_form
from zcu_lab.v2.support.measure.analyze_results import fit_quality_summary

if TYPE_CHECKING:
    from zcu_tools.plotting.plots import Plots

GERunResult: TypeAlias = RunRecord[GE_Cfg, GE_Result]


@dataclass
class GEAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="Initial State")
    ] = "ground"
    # ``backend`` selects the primary rotation/threshold fit. Post-analysis uses
    # the resulting centres and does not choose or run another fit backend.
    backend: Annotated[Literal["pca", "center"], ParamMeta(label="Backend")] = "pca"
    logscale: Annotated[bool, ParamMeta(label="Log Scale")] = False
    align_t1: Annotated[bool, ParamMeta(label="Align T1")] = True
    length_ratio: Annotated[
        float | None, ParamMeta(label="Length Ratio", decimals=4)
    ] = None


class GESummary(BaseModel):
    """Analysis summary in native units; null retains an unavailable estimate."""

    fidelity: float | None = Field(
        description="Owner fidelity in dimensionless; null retains an unavailable estimate."
    )

    theta: float | None = Field(
        description="Owner theta in rad; null retains an unavailable estimate."
    )

    threshold: float | None = Field(
        description="Owner threshold in ADC units; null retains an unavailable estimate."
    )


class GEPostSummary(BaseModel):
    """Analysis summary in native units; null retains an unavailable estimate."""

    ge_radius: float | None = Field(
        description="Owner ge_radius in ADC units; null retains an unavailable estimate."
    )

    confusion: list[list[float]] = Field(description="Owner confusion values.")


@dataclass(frozen=True)
class GEAnalyzeResult(GEAnalysis, AnalyzeResultBase):
    """Fit calibration, with a JSON-safe GUI projection of the core result."""

    def summary(self) -> GESummary:
        """Return the named scalar summary consumed by recipes, in native units."""
        return GESummary(
            fidelity=self.fidelity,
            theta=self.theta,
            threshold=self.threshold,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            **self.summary().model_dump(),
            "initial_state": self.initial_state,
            "ge_s": self.ge_s,
            "init_pops": self.init_pops.tolist(),
            "fit_quality": fit_quality_summary(self.fit_quality),
        }


@dataclass(frozen=True)
class GEPostAnalyzeResult(PostAnalyzeResultBase):
    """Radius in normalized ADC units, 3x3 matrix and dimensionless condition."""

    ge_radius: float
    confusion: list[list[float]]
    condition_number: float

    def summary(self) -> GEPostSummary:
        """Return the named radius and confusion summary in native ADC units."""
        return GEPostSummary(ge_radius=self.ge_radius, confusion=self.confusion)

    def to_summary_dict(self) -> dict[str, object]:
        return {
            **self.summary().model_dump(),
            "condition_number": self.condition_number,
        }


def ge_calibration_review_note(primary: GEAnalysis) -> str | None:
    """Explain manual selection for single/mux GE model-derived calibrations.

    Joint histogram r2 must be >=0.9, normalized RMS <=0.1 and relative width
    error <=0.5. These dimensionless gates also guard empirical assignment and
    centers because a poor model is not calibration evidence. Required missing /
    nonfinite metrics or their invalid issues prevent preselection. Unrelated
    fixed-zero parameter issues do not. None means these review gates passed,
    not that preparation or pure-state readout fidelity has been established.
    """
    quality = None if primary.fit_quality is None else primary.fit_quality.get("joint")
    if quality is None:
        return "Review required: joint histogram quality unavailable"
    reasons = [
        f"joint {issue.path}: {issue.reason}"
        for issue in quality.invalid
        if issue.path
        in {"r2", "normalized_residual_rms", "relative_parameter_errors.s"}
    ]
    if quality.r2 is None or not math.isfinite(quality.r2):
        reasons.append("joint r² unavailable")
    elif quality.r2 < 0.9:
        reasons.append(f"joint r²={quality.r2:.4g} <0.9")
    rms = quality.normalized_residual_rms
    if rms is None or not math.isfinite(rms):
        reasons.append("joint normalized RMS unavailable")
    elif rms > 0.1:
        reasons.append(f"joint normalized RMS={rms:.4g} >0.1")
    width_error = quality.relative_parameter_errors.get("s")
    if width_error is None or not math.isfinite(width_error):
        reasons.append("relative width error unavailable")
    elif width_error > 0.5:
        reasons.append(f"relative width error={width_error:.4g} >0.5")
    return "Review required: " + "; ".join(reasons) if reasons else None


def ge_post_review_note(primary: GEAnalysis, condition_number: float) -> str | None:
    """Guard radius/confusion by primary GE quality and finite condition <=10.

    The condition is dimensionless. Manual draft selection remains possible;
    None means the review gates passed, not that the correction is reliable.
    """
    primary_note = ge_calibration_review_note(primary)
    reasons = [] if primary_note is None else [primary_note]
    if not math.isfinite(condition_number):
        reasons.append("post condition number unavailable")
    elif condition_number > 10:
        reasons.append(f"post condition number={condition_number:.4g} >10")
    return "; ".join(reasons) if reasons else None


def _analyze(
    req: AnalyzeRequest[GERunResult, GEAnalyzeParams], *, plots: Plots
) -> GEAnalyzeResult:
    params = req.analyze_params
    analysis = GE_Exp().analyze(
        req.run_result,
        GEAnalyzeOptions(
            initial_state=params.initial_state,
            backend=params.backend,
            logscale=params.logscale,
            align_t1=params.align_t1,
            length_ratio=params.length_ratio,
        ),
        plots=plots,
    )
    return GEAnalyzeResult(
        initial_state=analysis.initial_state,
        fidelity=analysis.fidelity,
        theta=analysis.theta,
        threshold=analysis.threshold,
        ge_s=analysis.ge_s,
        g_center=analysis.g_center,
        e_center=analysis.e_center,
        init_pops=analysis.init_pops,
        fit_quality=analysis.fit_quality,
    )


def _post_analyze(
    req: PostAnalyzeRequest[GERunResult, GEAnalyzeResult, None],
    *,
    plots: Plots,
) -> GEPostAnalyzeResult:
    analysis = GE_Exp().post_analyze(
        req.run_result, req.analyze_result, GEPostAnalyzeOptions(), plots=plots
    )
    return GEPostAnalyzeResult(
        ge_radius=analysis.confusion.radius,
        confusion=analysis.confusion.matrix.tolist(),
        condition_number=analysis.confusion.condition_number,
    )


def _writeback(
    req: WritebackRequest[GERunResult, GEAnalyzeResult],
) -> Sequence[WritebackItem]:
    result = req.analyze_result
    result.validate_calibration()
    note = ge_calibration_review_note(result)
    # AcceptedWrite owns native complex conversion; producers keep numbers.
    return [
        ParameterWriteback(
            target_name="$qubit.ext.readout_cal.fidelity",
            description="Prepared-label assignment fidelity",
            proposed_value=result.fidelity,
            selected=note is None,
            note=note,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.readout_cal.sigma",
            description="Single-shot IQ cluster width (ADC units)",
            proposed_value=result.ge_s,
            selected=note is None,
            note=note,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.readout_cal.g_center",
            description="Single-shot |g> IQ cluster centre (complex)",
            proposed_value=result.g_center,
            selected=note is None,
            note=note,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.readout_cal.e_center",
            description="Single-shot |e> IQ cluster centre (complex)",
            proposed_value=result.e_center,
            selected=note is None,
            note=note,
        ),
    ]


def _post_writeback(
    req: PostWritebackRequest[GERunResult, GEAnalyzeResult, GEPostAnalyzeResult],
) -> Sequence[WritebackItem]:
    result = req.post_analyze_result
    note = ge_post_review_note(req.analyze_result, result.condition_number)
    return [
        ParameterWriteback(
            target_name="$qubit.ext.readout_cal.radius",
            description="Single-shot classification radius",
            proposed_value=result.ge_radius,
            selected=note is None,
            note=note,
        ),
        ParameterWriteback(
            target_name="$qubit.ext.readout_cal.confusion_matrix",
            description="Single-shot 3x3 confusion matrix (prepared->measured)",
            proposed_value=result.confusion,
            selected=note is None,
            note=note,
        ),
    ]


def _run(cfg: GE_Cfg, *, context: RunContext) -> RunRecord[GE_Cfg, GE_Result]:
    return RunRecord(cfg=cfg, result=GE_Exp().run(cfg, context=context))


DEFINITION = Adapter(
    spec=GE_Exp.AXES_SPEC,
    make_form=make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_analyze,
        options=lambda source, env: GEAnalyzeParams(),
        writeback=_writeback,
    ),
    post_analysis=PostAnalysis(fn=_post_analyze, writeback=_post_writeback),
)
