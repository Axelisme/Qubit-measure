"""Squared-rate sine/cosine regression with maximum-phase uncertainty."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .quality import FitQuality, compute_fit_quality


@dataclass(frozen=True)
class PhaseModulationPrediction:
    """Resolved maximum and squared-rate prediction on a dense degree axis.

    phase_degrees: float64 (721,) axis from -180 to 180 degrees.
    squared_rates: float64 (721,) model values in squared-rate units.
    maximum_phase_deg: atan2(C,B) in [-180,180] degrees.
    """

    phase_degrees: NDArray[np.float64]
    squared_rates: NDArray[np.float64]
    maximum_phase_deg: float


@dataclass(frozen=True)
class PhaseModulationFit:
    """Regression of squared rate = A + B cos(phase) + C sin(phase).

    reason: None on success, otherwise the unavailable-phase explanation.
    fit_quality: residual/parameter diagnostics, or None before regression.
    coefficients: float64 (3,) in A,B,C order, in squared-rate units, or None
        when regression cannot be formed.
    covariance: float64 (3,3) residual-scaled covariance in A,B,C order, with
        squared squared-rate units, or None when regression cannot be formed.
    phase_variance_rad2: propagated atan2 variance in radians², or None when
        the maximum is unresolved.
    phase_error_deg: 1σ maximum-phase error in degrees, or None on failure.
    prediction: resolved dense model and maximum, or None when unresolved.
    """

    reason: str | None = None
    fit_quality: FitQuality | None = None
    coefficients: NDArray[np.float64] | None = None
    covariance: NDArray[np.float64] | None = None
    phase_variance_rad2: float | None = None
    phase_error_deg: float | None = None
    prediction: PhaseModulationPrediction | None = None


def fit_phase_modulation(
    phases: NDArray[np.float64], squared_rates: NDArray[np.float64]
) -> PhaseModulationFit:
    """Regress squared rates on sine/cosine and propagate maximum-phase error.

    phases and squared_rates are matching float64 (N,) arrays, in degrees and
    squared-rate units respectively. Caller selects identifiable rate slices.
    More than three slices and full-rank sine/cosine coverage are required.
    Covariance uses ordinary least squares with residual/(N-3) scaling.
    Rank deficiency, linear algebra failure, roundoff-zero modulation, nonfinite
    uncertainty or nonpositive maximum yield an unavailable result with reason.
    No contrast, sampling or residual acceptance threshold is applied. Other
    array/shape errors propagate. Inputs are not modified.
    """
    if len(phases) <= 3:
        return PhaseModulationFit(
            reason="Need more than three identifiable phase slices"
        )
    radians = np.deg2rad(phases)
    design = np.column_stack((np.ones_like(radians), np.cos(radians), np.sin(radians)))
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, squared_rates, rcond=None)
        if rank != 3:
            return PhaseModulationFit(
                reason="Degenerate phase coverage: sine/cosine design is rank deficient"
            )
        coefficients = np.asarray(coefficients, dtype=np.float64)
        fitted = design @ coefficients
        covariance = (
            np.linalg.inv(design.T @ design)
            * np.sum((squared_rates - fitted) ** 2)
            / (len(phases) - 3)
        )
    except np.linalg.LinAlgError as exc:
        return PhaseModulationFit(reason=f"Interference fit failed: {exc}")
    quality = compute_fit_quality(
        squared_rates,
        fitted,
        dict(zip(("A", "B", "C"), coefficients, strict=True)),
        covariance,
    )
    a, b, c = coefficients
    modulation = np.hypot(b, c)
    # Only reject a zero at floating-point roundoff, not a physical contrast gate.
    numerical_zero = 8 * np.finfo(float).eps * np.linalg.norm(coefficients)
    if modulation <= numerical_zero:
        return PhaseModulationFit(
            reason="Zero rate² modulation: constructive phase undefined",
            fit_quality=quality,
            coefficients=coefficients,
            covariance=covariance,
        )
    gradient = np.array([0.0, -c, b]) / modulation**2
    phase_variance = float(gradient @ covariance @ gradient)
    if (
        not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(covariance))
        or not np.isfinite(phase_variance)
        or phase_variance < 0
        or a + modulation <= 0
    ):
        return PhaseModulationFit(
            reason="Nonfinite interference fit or unavailable phase uncertainty",
            fit_quality=quality,
            coefficients=coefficients,
            covariance=covariance,
        )
    phase = float(np.rad2deg(np.arctan2(c, b)))
    dense = np.linspace(-180.0, 180.0, 721)
    curve = a + b * np.cos(np.deg2rad(dense)) + c * np.sin(np.deg2rad(dense))
    return PhaseModulationFit(
        None,
        quality,
        coefficients,
        covariance,
        phase_variance,
        float(np.rad2deg(np.sqrt(phase_variance))),
        PhaseModulationPrediction(dense, curve, phase),
    )
