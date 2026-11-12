"""Weighted linear regression after degree-phase unwrap."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quality import FitQuality, compute_fit_quality


@dataclass(frozen=True)
class PhaseLineFit:
    """Unwrapped line with absolute-variance covariance, or unavailable fields.

    unwrapped_deg: float64 (N,) phases unwrapped with period 360°, or None.
    fit_curve: float64 (N,) predicted phases at the original n, or None.
    intercept_deg: fitted phase at n=0 in degrees, or None.
    slope_deg: fitted phase change per n unit in degrees, or None.
    slope_error_deg: absolute-variance 1σ slope error in degrees/n unit, or None.
    covariance: float64 (2,2) in intercept,slope order, with product units
        (degree², degree²/n, degree²/n²), or None on failure.
    fit_quality: residual and parameter-error diagnostics, or None on failure.
    reason: None on success; otherwise the reason all numeric fields are None.
    """

    unwrapped_deg: NDArray[np.float64] | None = None
    fit_curve: NDArray[np.float64] | None = None
    intercept_deg: float | None = None
    slope_deg: float | None = None
    slope_error_deg: float | None = None
    covariance: NDArray[np.float64] | None = None
    fit_quality: FitQuality | None = None
    reason: str | None = None


def fit_phase_line(
    n: NDArray[np.float64],
    phase_degrees: NDArray[np.float64],
    variances: NDArray[np.float64],
) -> PhaseLineFit:
    """Unwrap degree phases, then fit intercept+slope*n with absolute weights.

    Inputs are matching float64 (N,) arrays in acquisition order. n is the
    coordinate axis, phase_degrees contains complete finite degree phases,
    and variances contains their variances in degree². Caller resolves missing
    phases before calling. At least two samples with two distinct n values
    are required. Inputs are not modified. Unwrap uses period 360° and cannot
    infer steps exceeding a half turn. An exact adjacent 180° step is ambiguous.
    Nonpositive/nonfinite variances, half-turn ambiguity, rank deficiency or
    nonfinite fitted values/covariance return None fields and a reason.
    Covariance is (XᵀWX)⁻¹ without residual scaling. Array/linear algebra errors
    otherwise propagate; no correction sign or calibration policy is applied.
    """
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
        return PhaseLineFit(reason="Unavailable positive count-derived phase variance")
    y = np.unwrap(phase_degrees, period=360)
    if np.any(np.abs(np.diff(y)) == 180):
        return PhaseLineFit(reason="Adjacent half-turn phase step has ambiguous unwrap")
    design = np.column_stack((np.ones_like(n), n))
    weighted = design / np.sqrt(variances[:, None])
    if np.linalg.matrix_rank(weighted) != 2:
        return PhaseLineFit(reason="Unidentifiable linear model")
    covariance = np.asarray(np.linalg.inv(weighted.T @ weighted), dtype=np.float64)
    intercept, slope = np.linalg.lstsq(weighted, y / np.sqrt(variances), rcond=None)[0]
    curve = intercept + slope * n
    if (
        not np.all(np.isfinite([slope, intercept]))
        or not np.all(np.isfinite(covariance))
        or not np.all(np.isfinite(curve))
    ):
        return PhaseLineFit(reason="Nonfinite weighted fit or covariance uncertainty")
    quality = compute_fit_quality(
        y,
        curve,
        {"intercept_deg": float(intercept), "slope_deg": float(slope)},
        covariance,
    )
    return PhaseLineFit(
        y,
        curve,
        float(intercept),
        float(slope),
        float(np.sqrt(covariance[1, 1])),
        covariance,
        quality,
    )
