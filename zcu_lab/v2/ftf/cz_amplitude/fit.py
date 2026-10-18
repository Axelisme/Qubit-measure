"""Zero-phase, first-positive-cycle CZ return policy from decision 0089."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality, fitcos


@dataclass(frozen=True)
class CZReturnFit:
    """Single-frequency diagnostics and an optional measured first-return scale.

    scale is the first positive period only when reason=None; otherwise no
    calibration is available. period, offset and contrast retain finite fitted
    diagnostics even if coverage prevents a proposal. error is the period's
    finite standard error, not an acceptance threshold. curve matches the full
    scale axis, with NaN outside the complete prefix used by the optimizer.
    fit_quality describes those actual observations and transformed covariance.
    No free phase, contrast/sigma, chi-square or fidelity gate is used.
    """

    scale: float | None
    period: float | None
    error: float | None
    offset: float | None
    contrast: float | None
    curve: NDArray[np.float64]
    fit_quality: FitQuality | None
    reason: str | None


def fit_return(
    scales: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    complete: NDArray[np.bool_],
) -> CZReturnFit:
    """Fit complete prefix P00 to offset + contrast*cos(2π*scale/period).

    Inputs are classify_return's finite ascending uniform source scales, P00
    including incomplete NaNs, and its completion mask. No source is modified.
    Require measured scale=0 and every requested point through the first positive
    return. A cancelled suffix may be absent; a missing interior scale ends the
    prefix. Numeric failure, zero contrast, rank deficiency or nonfinite standard
    errors produce a reason, never a default amplitude. A finite diagnostic fit
    beyond the measured span is retained but cannot supply a calibration.
    """
    answer = CZReturnFit(
        None, None, None, None, None, np.full_like(scales, np.nan), None, None
    )
    if scales[0] != 0 or not complete[0]:
        return replace(
            answer, reason="Measured scale=0 is required for first-cycle return"
        )
    missing = np.flatnonzero(~complete)
    count = int(missing[0]) if len(missing) else len(scales)
    xs, ys = scales[:count], probabilities[:count]
    if count <= 3:
        return replace(
            answer,
            reason="Insufficient complete scales for a three-parameter return fit",
        )
    if np.ptp(ys) <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(ys)))):
        return replace(answer, reason="Zero return contrast; period is degenerate")
    return _fit_complete_prefix(xs, ys, answer)


def _fit_complete_prefix(
    xs: NDArray[np.float64], ys: NDArray[np.float64], answer: CZReturnFit
) -> CZReturnFit:
    """Fit the contiguous valid observations, transform uncertainty and check coverage."""
    try:
        parameters, covariance = fitcos(xs, ys, fixedparams=[None, None, None, 0.0])
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return replace(answer, reason=f"Zero-phase return fit failed: {exc}")
    offset, amplitude, frequency, phase = map(float, parameters)
    if not np.all(np.isfinite(parameters)) or frequency <= 0:
        return replace(
            answer,
            reason="Return fit has nonfinite parameters or nonpositive frequency",
        )
    # fitcos canonicalizes negative amplitudes with a 180° shift. Restore the
    # signed zero-phase coefficient; phase was fixed, never a fitted parameter.
    sign = float(np.cos(np.deg2rad(phase)))
    contrast, period = sign * amplitude, 1.0 / frequency
    jacobian = np.diag([1.0, sign, -1.0 / frequency**2])
    transformed = jacobian @ covariance[:3, :3] @ jacobian.T
    curve = offset + contrast * np.cos(2 * np.pi * frequency * xs)
    quality = compute_fit_quality(
        ys,
        curve,
        {"offset": offset, "contrast": contrast, "period": period},
        transformed,
    )
    full_curve = answer.curve.copy()
    full_curve[: len(xs)] = curve
    answer = replace(
        answer,
        period=period,
        offset=offset,
        contrast=contrast,
        curve=full_curve,
        fit_quality=quality,
    )
    if not np.all(np.isfinite(transformed)) or np.any(np.diag(transformed) < 0):
        return replace(
            answer,
            reason="Return fit uncertainty is not finite or covariance is invalid",
        )
    errors = np.sqrt(np.diag(transformed))
    answer = replace(answer, error=float(errors[2]))
    derivatives = np.column_stack(
        (
            np.ones_like(xs),
            np.cos(2 * np.pi * frequency * xs),
            contrast
            * 2
            * np.pi
            * frequency**2
            * xs
            * np.sin(2 * np.pi * frequency * xs),
        )
    )
    if contrast == 0 or np.linalg.matrix_rank(derivatives) < 3:
        return replace(
            answer, reason="Return model is degenerate; period is unidentifiable"
        )
    if period > xs[-1]:
        return replace(
            answer,
            reason="First positive return is outside the complete measured span; no extrapolation",
        )
    return replace(answer, scale=period)
