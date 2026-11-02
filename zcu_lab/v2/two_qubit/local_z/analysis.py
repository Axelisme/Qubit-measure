"""Saved-source local-Z quadratures and thermal ensemble phase diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality, compute_fit_quality

from .._phase import QuadraturePhase


@dataclass(frozen=True)
class LocalZTargetFit:
    """One target's complete-row weighted linear model, phase_deg = intercept + slope*n.

    unwrapped_deg/fit_curve have Nn entries, or None when any phase is missing.
    slope_deg is logical Bloch accrual per CZ; correction_deg is its negative,
    never rewrapped. intercept_deg is the preparation/projection offset.
    slope_error_deg is the 1σ error from the count-propagated fit covariance.
    fit_quality holds residual/relative uncertainty diagnostics without a gate.
    Unavailable fits retain None values and reason, never a substituted zero.
    """

    unwrapped_deg: NDArray[np.float64] | None = None
    fit_curve: NDArray[np.float64] | None = None
    intercept_deg: float | None = None
    slope_deg: float | None = None
    correction_deg: float | None = None
    slope_error_deg: float | None = None
    fit_quality: FitQuality | None = None
    reason: str | None = None


@dataclass(frozen=True)
class LocalZFit:
    """One target's phase-line summary, in degrees and degrees per gate.

    correction_deg is minus slope_deg without wrapping. intercept_deg is the
    preparation/projection offset, slope_error_deg is its count-derived 1σ
    uncertainty. Unavailable fits keep None and reason. fit_quality contains
    fixed-size residual and parameter-error diagnostics, not a quality gate.
    """

    intercept_deg: float | None
    slope_deg: float | None
    correction_deg: float | None
    slope_error_deg: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class LocalZAnalysis:
    """Fixed F1/F2 phase-line summaries in target order; no scan arrays."""

    fits: tuple[LocalZFit, ...]


def summarize_fit(fit: LocalZTargetFit) -> LocalZFit:
    """Keep calibration values and fit diagnostics; leave curves for plotting."""
    return LocalZFit(
        fit.intercept_deg,
        fit.slope_deg,
        fit.correction_deg,
        fit.slope_error_deg,
        fit.fit_quality,
        fit.reason,
    )


def fit_target(
    n: NDArray[np.float64],
    points: tuple[QuadraturePhase, ...],
    variances_deg2: NDArray[np.float64],
) -> LocalZTargetFit:
    """Unwrap a complete phase row and fit intercept+slope*n with count weights.

    n and variances_deg2 match points, with at least three distinct n values.
    Missing phases, ambiguous half turns, or nonpositive/nonfinite variances
    return an unavailable fit with a reason. Covariance is (XᵀWX)⁻¹, not
    residual-rescaled. Correction is minus slope without rewrapping.
    """
    if any(p.logical_phase_deg is None for p in points):
        return LocalZTargetFit(reason="Incomplete phase row")
    if not np.all(np.isfinite(variances_deg2)) or np.any(variances_deg2 <= 0):
        return LocalZTargetFit(
            reason="Unavailable positive count-derived phase variance"
        )
    y = np.unwrap(
        np.array([p.logical_phase_deg for p in points], dtype=np.float64), period=360
    )
    if np.any(np.abs(np.diff(y)) == 180):
        return LocalZTargetFit(
            reason="Adjacent half-turn phase step has ambiguous unwrap"
        )
    design = np.column_stack((np.ones_like(n), n))
    weighted = design / np.sqrt(variances_deg2[:, None])
    if np.linalg.matrix_rank(weighted) != 2:
        return LocalZTargetFit(reason="Unidentifiable linear model")
    covariance = np.asarray(np.linalg.inv(weighted.T @ weighted), dtype=np.float64)
    intercept, slope = np.linalg.lstsq(
        weighted, y / np.sqrt(variances_deg2), rcond=None
    )[0]
    curve = intercept + slope * n
    if (
        not np.all(np.isfinite([slope, intercept]))
        or not np.all(np.isfinite(covariance))
        or not np.all(np.isfinite(curve))
    ):
        return LocalZTargetFit(
            reason="Nonfinite weighted fit or covariance uncertainty"
        )
    quality = compute_fit_quality(
        y,
        curve,
        {"intercept_deg": float(intercept), "slope_deg": float(slope)},
        covariance,
    )
    return LocalZTargetFit(
        y,
        curve,
        float(intercept),
        float(slope),
        -float(slope),
        float(np.sqrt(covariance[1, 1])),
        quality,
    )
