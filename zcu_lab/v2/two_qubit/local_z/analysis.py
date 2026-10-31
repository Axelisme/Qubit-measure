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
    slope_error_deg is the 1σ count-propagated weighted-fit covariance.
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
class LocalZAnalysis:
    """F1/F2 classified phase diagnostics and complete-row weighted linear fits.

    phases is (2,Nn); quadratures is (2,2,Nn) sigma, and acquired_shots is
    the projection's terminal total. fits is F1 then F2; gate_counts is the n axis.
    herald_matrix is the full herald distribution or None for ordinary acquisition.
    pass_fraction is herald (g,g) acceptance, not ground-state purity.
    """

    phases: tuple[tuple[QuadraturePhase, ...], ...]
    quadratures: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    gate_counts: tuple[int, ...]
    fits: tuple[LocalZTargetFit, ...]
    herald_matrix: NDArray[np.float64] | None = None
    pass_fraction: NDArray[np.float64] | None = None


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
    covariance = np.linalg.inv(weighted.T @ weighted)
    intercept, slope = np.linalg.lstsq(
        weighted, y / np.sqrt(variances_deg2), rcond=None
    )[0]
    curve = intercept + slope * n
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
