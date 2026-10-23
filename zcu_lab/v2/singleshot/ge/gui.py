from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    PostAnalyzeRequest,
    PostAnalyzeResultBase,
    PostWritebackRequest,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.singleshot.ge.core import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalyzeOptions,
)
from zcu_lab.v2.singleshot.ge.form import make_form

if TYPE_CHECKING:
    from zcu_tools.gui.form_adapter import Form, FormContext
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


@dataclass
class GEPostAnalyzeParams:
    """The GE confusion diagnostic has no independent operator parameters."""


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


class GEAdapter(
    BaseAdapter[
        GE_Cfg, GERunResult, GEAnalyzeResult, GEAnalyzeParams, GEPostAnalyzeResult
    ]
):
    experiment_tag: ClassVar[str | None] = GE_Exp.persistence_tag()

    @staticmethod
    def make_form(ctx: FormContext) -> Form[GE_Cfg]:
        """Build this adapter's definition using captured Open/Reset sources."""
        return make_form(ctx)

    exp_cls = GE_Exp
    # FIT primary analysis + a confusion-diagnostic post-analysis layer.
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.FIT, post_analysis=True, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    def run(
        self,
        req: RunRequest,
        cfg: GE_Cfg,
        *,
        context: RunContext,
    ) -> GERunResult:
        del req
        result = GE_Exp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def analyze(
        self, req: AnalyzeRequest[GERunResult, GEAnalyzeParams], *, plots: Plots
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

    def get_post_analyze_params(
        self, analyze_result: GEAnalyzeResult, ctx: SessionEnv
    ) -> GEPostAnalyzeParams:
        del analyze_result, ctx
        return GEPostAnalyzeParams()

    def post_analyze(
        self,
        req: PostAnalyzeRequest[GERunResult, GEAnalyzeResult, GEPostAnalyzeParams],
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

    def get_writeback_items(
        self, req: WritebackRequest[GERunResult, GEAnalyzeResult]
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

    def get_post_writeback_items(
        self,
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
