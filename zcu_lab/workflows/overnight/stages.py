"""Fixed-cfg overnight acquisitions, native saving and lifetime acceptance."""

from dataclasses import dataclass

import numpy as np
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Failed

from zcu_lab.v2.support.lifetime_models import (
    LifetimeModelRecord,
    show_lifetime_comparison,
)
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Exp
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoCfg, T2EchoExp
from zcu_lab.workflows.common.analysis import (
    DecayFit,
    analyze_t1,
    compare_decay_fit,
    lifetime_rejection,
)
from zcu_lab.workflows.common.calibration_results import StageStatus
from zcu_lab.workflows.common.gates import Gate, gate
from zcu_lab.workflows.common.stages import NativeOutput, StageEnv, acquire_saved
from zcu_lab.workflows.overnight.analysis import analyze_t2echo


@dataclass(frozen=True, kw_only=True)
class FitSettings:
    """Acceptance and model diagnostics shared by this run's T1/Echo fits.

    max_rel_err and max_fit_residual are strict dimensionless upper bounds.
    min_lifetime_points counts fitted delays within [0, lifetime_us].
    max_model_window_factor only controls diagnostic model discard, not acceptance.
    """

    max_rel_err: float
    max_fit_residual: float
    min_lifetime_points: int
    max_model_window_factor: float


@dataclass(frozen=True, kw_only=True)
class T1Result:
    """T1 status and accepted lifetime/error in us, absent unless accepted.

    reason is None on acceptance, stopped on partial, otherwise the failure or
    failed gates. file references saved Completed raw, including partials and
    rejected fits; Failed acquisitions have no file. models holds both diagnostic
    fits after primary-fit success. gates holds numeric acceptance comparisons.
    Unexpected errors propagate without a result; saved raw remains on disk.
    """

    status: StageStatus
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    file: str | None = None
    models: tuple[LifetimeModelRecord, ...] = ()
    gates: tuple[Gate, ...] = ()


@dataclass(frozen=True, kw_only=True)
class EchoResult:
    """Echo status and accepted lifetime/error in us, absent unless accepted.

    reason is None on acceptance, stopped on partial, otherwise the failure or
    failed gates. file references saved Completed raw, including partials and
    rejected fits; Failed acquisitions have no file. models holds both diagnostic
    fits after primary-fit success. gates holds numeric acceptance comparisons.
    Unexpected errors propagate without a result; saved raw remains on disk.
    """

    status: StageStatus
    reason: str | None = None
    lifetime_us: float | None = None
    error_us: float | None = None
    file: str | None = None
    models: tuple[LifetimeModelRecord, ...] = ()
    gates: tuple[Gate, ...] = ()


def _acceptance(
    fit: DecayFit, settings: FitSettings, name: str
) -> tuple[str | None, tuple[Gate, ...]]:
    gates = (
        gate("positive_lifetime_us", ">", 0, fit.lifetime_us),
        gate("nonnegative_error_us", ">=", 0, fit.error_us),
        gate(
            "relative_error",
            "<",
            settings.max_rel_err,
            fit.error_us / fit.lifetime_us if fit.lifetime_us > 0 else None,
        ),
        gate(
            "normalized_residual_rms",
            "<",
            settings.max_fit_residual,
            fit.quality.normalized_residual_rms,
        ),
        gate("invalid_quality_count", "==", 0, len(fit.quality.invalid)),
        gate(
            "points_within_lifetime",
            ">=",
            settings.min_lifetime_points,
            int(np.count_nonzero((fit.times >= 0) & (fit.times <= fit.lifetime_us))),
        ),
    )
    rejection = lifetime_rejection(
        fit.quality, fit.times, fit.lifetime_us, settings.min_lifetime_points
    )
    reasons = [rejection] if rejection is not None else []
    failed_comparison = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!="}
    reasons.extend(
        f"{item.name} {item.actual:.9g} {failed_comparison[item.comparison]} {item.threshold:.9g}"
        if item.actual is not None
        else f"{item.name} unavailable (requires {item.comparison} {item.threshold:.9g})"
        for item in gates
        if not item.passed
    )
    reason = f"{name} fit rejected: {'; '.join(reasons)}" if reasons else None
    return reason, gates


async def run_t1_stage(
    scope: Scope, env: StageEnv, cfg: T1Cfg, settings: FitSettings
) -> T1Result:
    """Acquire the fixed T1 cfg, save t1.h5, fit and apply captured settings.

    cfg supplies calibrated pulses, delay grid and averaging without feedback.
    env supplies plots, historical save identity and summary observation. Partial
    returns rejected/stopped without fitting; acquisition/expected fit failure
    returns failed. No point record or history is changed. Other errors propagate.
    """
    acquisition = await acquire_saved(
        scope,
        env,
        T1Exp().run,
        cfg,
        name="T1",
        output=NativeOutput(filename="t1.h5", spec=T1Exp.AXES_SPEC),
    )
    if acquisition.result is None:
        return T1Result(
            status="rejected" if acquisition.status == "partial" else "failed",
            reason=acquisition.reason,
            file=acquisition.file,
        )
    fit = analyze_t1(acquisition.result)
    if isinstance(fit, Failed):
        return T1Result(status="failed", reason=fit.reason, file=acquisition.file)
    comparison = compare_decay_fit(
        fit, t1=True, max_window_factor=settings.max_model_window_factor
    )
    show_lifetime_comparison(
        env.plots, "t1-models", fit.times, fit.observed, comparison
    )
    reason, gates = _acceptance(fit, settings, "T1")
    return T1Result(
        status="accepted" if reason is None else "rejected",
        reason=reason,
        lifetime_us=fit.lifetime_us if reason is None else None,
        error_us=fit.error_us if reason is None else None,
        file=acquisition.file,
        models=comparison.records,
        gates=gates,
    )


async def run_echo_stage(
    scope: Scope, env: StageEnv, cfg: T2EchoCfg, settings: FitSettings
) -> EchoResult:
    """Acquire the fixed zero-detune Echo cfg, save t2echo.h5, fit and gate.

    cfg supplies calibrated pulses, delay grid, relaxation and averaging. settings
    supplies captured acceptance and model diagnostics; env supplies mechanical
    services. Partial returns rejected/stopped without fitting. Acquisition or
    expected fit failure returns failed; other errors propagate. This stage does
    not change the point record, cancellation status or history.
    """
    acquisition = await acquire_saved(
        scope,
        env,
        T2EchoExp().run,
        cfg,
        name="Echo",
        output=NativeOutput(filename="t2echo.h5", spec=T2EchoExp.AXES_SPEC),
    )
    if acquisition.result is None:
        return EchoResult(
            status="rejected" if acquisition.status == "partial" else "failed",
            reason=acquisition.reason,
            file=acquisition.file,
        )
    fit = analyze_t2echo(acquisition.result)
    if isinstance(fit, Failed):
        return EchoResult(status="failed", reason=fit.reason, file=acquisition.file)
    comparison = compare_decay_fit(
        fit, t1=False, max_window_factor=settings.max_model_window_factor
    )
    show_lifetime_comparison(
        env.plots, "t2echo-models", fit.times, fit.observed, comparison
    )
    reason, gates = _acceptance(fit, settings, "T2 Echo")
    return EchoResult(
        status="accepted" if reason is None else "rejected",
        reason=reason,
        lifetime_us=fit.lifetime_us if reason is None else None,
        error_us=fit.error_us if reason is None else None,
        file=acquisition.file,
        models=comparison.records,
        gates=gates,
    )
