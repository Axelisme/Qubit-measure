from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.cfg_model import CircleClassifier
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    ParamMeta,
    PostAnalysis,
    PostAnalyzeRequest,
    PostWritebackRequest,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.writeback import calibrated_module_writeback
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2.singleshot.ge.core import (
    GE_Cfg,
    GE_Exp,
    GE_Result,
    GEAnalysis,
    GEAnalyzeOptions,
    GEPostAnalysis,
    GEPostAnalyzeOptions,
)
from zcu_lab.v2.singleshot.ge.core import (
    ge_calibration_review_note as ge_calibration_review_note,
)
from zcu_lab.v2.singleshot.ge.core import (
    ge_post_review_note as ge_post_review_note,
)
from zcu_lab.v2.singleshot.ge.form import make_form

GERunResult: TypeAlias = RunRecord[GE_Cfg, GE_Result]


@dataclass
class GEAnalyzeParams:
    initial_state: Annotated[
        Literal["ground", "excited"], ParamMeta(label="State before π probe")
    ] = "ground"
    # ``backend`` selects the primary rotation/threshold fit. Post-analysis uses
    # the resulting centres and does not choose or run another fit backend.
    backend: Annotated[
        Literal["pca", "center"], ParamMeta(label="IQ rotation method")
    ] = "pca"
    logscale: Annotated[
        bool, ParamMeta(label="Log scale for projected shot counts")
    ] = False
    align_t1: Annotated[
        bool, ParamMeta(label="Share decay ratio across histogram fits")
    ] = True
    length_ratio: Annotated[
        float | None, ParamMeta(label="Readout length / T1 (a.u.)", decimals=4)
    ] = None


class GESummary(BaseModel):
    """Decode consumed scalar JSON fields; null retains an unavailable estimate."""

    fit_quality: dict[str, FitQuality] | None = Field(
        description="Owner joint, ground and excited fit diagnostics; null when unavailable."
    )
    fidelity: float | None = Field(
        description="Prepared-label assignment fidelity (fraction); null when unavailable."
    )

    theta: float | None = Field(
        description="IQ rotation angle (rad); null when unavailable."
    )

    threshold: float | None = Field(
        description="Rotated I threshold (a.u.); null when unavailable."
    )


class GEConfusionSummary(BaseModel):
    """Consumed post-calibration JSON fields; nonfinite estimates arrive as null."""

    radius: float | None = Field(
        description="Classification radius in ADC units; null means unavailable."
    )
    matrix: list[list[float | None]] = Field(
        description="Prepared-to-measured confusion fractions; nonfinite entries are null."
    )
    condition_number: float | None = Field(
        description="Dimensionless matrix condition number; null means unavailable."
    )


class GEPostSummary(BaseModel):
    """Decode the consumed fields of the core post-analysis JSON object."""

    confusion: GEConfusionSummary = Field(
        description="Radius, confusion matrix and condition diagnostic from the core."
    )


def _analyze(
    req: AnalyzeRequest[GERunResult, GEAnalyzeParams], *, plots: Plots
) -> GEAnalysis:
    params = req.analyze_params
    return GE_Exp().analyze(
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


def _post_analyze(
    req: PostAnalyzeRequest[GERunResult, GEAnalysis, None], *, plots: Plots
) -> GEPostAnalysis:
    return GE_Exp().post_analyze(
        req.run_result, req.analyze_result, GEPostAnalyzeOptions(), plots=plots
    )


def _writeback(
    req: WritebackRequest[GERunResult, GEAnalysis],
) -> Sequence[WritebackItem]:
    result = req.analyze_result
    result.validate_calibration()
    note = ge_calibration_review_note(result)
    if req.ctx.parameters is None or req.run_result.cfg is None:
        raise ValueError("GE writeback requires captured parameters and readout cfg")
    item = calibrated_module_writeback(
        req.ctx.parameters,
        req.run_result.cfg.readout,
        target="$resonator.module.readout",
        description="Single-shot ground/excited centers",
        field_updates={
            "classifier": CircleClassifier(
                g_center=result.g_center,
                e_center=result.e_center,
            )
        },
    )
    item.selected, item.note = note is None, note
    return [item]


def _post_writeback(
    req: PostWritebackRequest[GERunResult, GEAnalysis, GEPostAnalysis],
) -> Sequence[WritebackItem]:
    result = req.post_analyze_result.confusion
    note = ge_post_review_note(req.analyze_result, result.condition_number)
    if req.ctx.parameters is None or req.run_result.cfg is None:
        raise ValueError("GE writeback requires captured parameters and readout cfg")
    item = calibrated_module_writeback(
        req.ctx.parameters,
        req.run_result.cfg.readout,
        target="$resonator.module.readout",
        description="Single-shot centers, radius and prepared-to-measured confusion",
        field_updates={
            "classifier": CircleClassifier(
                g_center=req.analyze_result.g_center,
                e_center=req.analyze_result.e_center,
                radius=result.radius,
                confusion_matrix=result.matrix.tolist(),
            )
        },
    )
    item.selected, item.note = note is None, note
    return [item]


def _summary_lines(source: GERunResult) -> tuple[str, ...]:
    cfg = source.cfg
    if cfg is None:
        return ()
    return (f"Reps×Rounds: {cfg.shots}×1 per preparation",)


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
        summary_lines=_summary_lines,
    ),
    post_analysis=PostAnalysis(fn=_post_analyze, writeback=_post_writeback),
)
