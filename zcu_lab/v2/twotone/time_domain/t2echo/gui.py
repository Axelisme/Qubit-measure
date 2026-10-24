from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field
from zcu_tools.analysis.fitting import FitQuality
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    Adapter,
    Analysis,
    AnalyzeRequest,
    AnalyzeResultBase,
    ParameterWriteback,
    ParamMeta,
    SampleWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.plotting.plots import Plots

from zcu_lab.v2._support.measure.analyze_results import fit_quality_summary
from zcu_lab.v2.twotone.time_domain.t2echo.core import (
    T2EchoAnalyzeOptions,
    T2EchoCfg,
    T2EchoExp,
    T2EchoResult,
)

from .form import T2EchoInput, make_form

logger = logging.getLogger(__name__)

T2EchoRunResult: TypeAlias = RunRecord[T2EchoCfg, T2EchoResult]


@dataclass
class T2EchoAnalyzeParams:
    fit_method: Annotated[Literal["fringe", "decay"], ParamMeta(label="Fit method")] = (
        "decay"
    )
    fit_phase: Annotated[bool, ParamMeta(label="Fit phase offset (fringe only)")] = (
        False
    )


class T2EchoSummary(BaseModel):
    """Analysis summary in native units; null retains an unavailable estimate."""

    t2e: float | None = Field(
        description="Owner t2e in us; null retains an unavailable estimate."
    )

    t2e_err: float | None = Field(
        description="Owner t2e_err in us; null retains an unavailable estimate."
    )


@dataclass
class T2EchoAnalyzeResult(AnalyzeResultBase):
    t2e: float
    t2e_err: float
    fit_quality: Mapping[str, FitQuality] | None = None

    def summary(self) -> T2EchoSummary:
        """Return the named scalar summary consumed by recipes, in native units."""
        return T2EchoSummary(
            t2e=self.t2e,
            t2e_err=self.t2e_err,
        )

    def to_summary_dict(self) -> dict[str, object]:
        return {
            **self.summary().model_dump(),
            "fit_quality": fit_quality_summary(self.fit_quality),
        }


def coherence_review_note(
    t2: float, t2_err: float, quality: FitQuality | None
) -> str | None:
    """Explain why single/mux Echo or mux Ramsey T2 needs manual selection.

    Times/errors are in us. Initial selection requires positive finite T2,
    nonnegative finite error/T2 <=0.5, r2 >=0.9 and normalized residual RMS <=0.1.
    Missing/nonfinite required metrics or their invalid issues require review.
    Unrelated fixed-zero parameter issues do not reject the result. None means
    these review gates passed, not that a calibration is physically established.
    """
    reasons = _coherence_fit_review_reasons(quality)
    if not math.isfinite(t2) or t2 <= 0:
        reasons.append("T2 must be finite and positive")
    elif not math.isfinite(t2_err) or t2_err < 0:
        reasons.append("T2 error must be finite and nonnegative")
    elif t2_err / t2 > 0.5:
        reasons.append(f"T2 relative error={t2_err / t2:.4g} >0.5")
    return "Review required: " + "; ".join(reasons) if reasons else None


def _coherence_fit_review_reasons(quality: FitQuality | None) -> list[str]:
    if quality is None:
        return ["fit quality unavailable"]
    reasons = [
        f"{issue.path}: {issue.reason}"
        for issue in quality.invalid
        if issue.path
        in {"r2", "normalized_residual_rms", "relative_parameter_errors.decay_time"}
    ]
    if quality.r2 is None or not math.isfinite(quality.r2):
        reasons.append("r² unavailable")
    elif quality.r2 < 0.9:
        reasons.append(f"r²={quality.r2:.4g} <0.9")
    rms = quality.normalized_residual_rms
    if rms is None or not math.isfinite(rms):
        reasons.append("normalized RMS unavailable")
    elif rms > 0.1:
        reasons.append(f"normalized RMS={rms:.4g} >0.1")
    return reasons


def _T2Echo_make_form(ctx: FormContext) -> Form[T2EchoInput]:
    """Build this adapter's definition using captured Open/Reset sources."""
    return make_form(ctx)


def _T2Echo_analyze(
    req: AnalyzeRequest[T2EchoRunResult, T2EchoAnalyzeParams],
    *,
    plots: Plots,
) -> T2EchoAnalyzeResult:
    params = req.analyze_params
    analysis = T2EchoExp().analyze(
        req.run_result,
        T2EchoAnalyzeOptions(fit_method=params.fit_method, fit_phase=params.fit_phase),
        plots=plots,
    )
    return T2EchoAnalyzeResult(
        t2e=analysis.t2e,
        t2e_err=analysis.t2e_err,
        fit_quality=analysis.fit_quality,
    )


def _T2Echo_get_writeback_items(
    req: WritebackRequest[T2EchoRunResult, T2EchoAnalyzeResult],
) -> Sequence[WritebackItem]:
    """Propose fitted T2E in us and a sample for the Run's resolved $qubit.

    req supplies the result and its captured context; source.cfg may be None.
    WritebackService resolves the role and destination on req.ctx.parameters
    and rejects a missing capture or unbound role. Without a capture,
    retain parameter proposals only so loaded data can still be analyzed.
    With a capture, unbound $qubit raises RoleResolutionError. The service
    adds bias coordinates or omits the sample with a public reason. This
    Run leaves devices fixed; no live lookup, device access, frequency/module
    update or write occurs here.
    """
    result = req.analyze_result
    quality = None if result.fit_quality is None else result.fit_quality.get("fit")
    note = coherence_review_note(result.t2e, result.t2e_err, quality)
    items: list[WritebackItem] = [
        ParameterWriteback(
            target_name="$qubit.t2e",
            description="T2 Echo time (us)",
            proposed_value=result.t2e,
            selected=note is None,
            note=note,
        ),
    ]
    parameters = req.ctx.parameters
    if parameters is not None:
        items.append(
            SampleWriteback(
                coordinate_device="$qubit.flux_source",
                target_name=parameters.resolve_path("$qubit"),
                description="T2 Echo sample at the captured bias",
                proposed_value={
                    "t2e_us": result.t2e,
                    "t2e_err_us": result.t2e_err,
                },
                selected=note is None,
                note=note,
            )
        )
    return items


def _run(
    inp: T2EchoInput, *, context: RunContext
) -> RunRecord[T2EchoCfg, T2EchoResult]:
    """Convert fringes/step to MHz, acquire and record the actual detune cfg.

    Zero ratio disables detune. Nonzero ratio on a zero-step delay sweep fails
    before acquisition. The measured, hardware-rounded detune is logged in MHz.
    """
    if inp.detune_ratio == 0.0:
        detune = 0.0
    elif inp.length.step == 0.0:
        raise ValueError(
            "cannot apply a nonzero detune_ratio on a degenerate "
            "length sweep (expts < 2, step == 0)"
        )
    else:
        detune = inp.detune_ratio / inp.length.step
    cfg = T2EchoCfg(
        reps=inp.reps,
        rounds=inp.rounds,
        relax_delay=inp.relax_delay,
        reset=inp.reset,
        pi2_pulse=inp.pi2_pulse,
        pi_pulse=inp.pi_pulse,
        readout=inp.readout,
        length=inp.length,
        detune=detune,
    )
    result = T2EchoExp().run(cfg, context=context)
    logger.info("T2Echo true detune: %s MHz", result.true_activate_detune)
    return RunRecord(cfg=cfg, result=result)


DEFINITION = Adapter(
    spec=T2EchoExp.AXES_SPEC,
    make_form=_T2Echo_make_form,
    run=_run,
    guide=Path(__file__).with_name("guide.md"),
    analysis=Analysis(
        fn=_T2Echo_analyze,
        options=lambda source, env: T2EchoAnalyzeParams(),
        writeback=_T2Echo_get_writeback_items,
    ),
)
