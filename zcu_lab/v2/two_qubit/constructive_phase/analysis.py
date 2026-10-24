"""Saved-source constructive-phase selection under the rate-squared model."""

from dataclasses import dataclass

import numpy as np
from zcu_tools.analysis.fitting import compute_fit_quality
from zcu_tools.analysis.fitting.quality import FitQuality
from zcu_tools.plotting.plots import Plots

from .._rabi_scan import (
    RabiAnalysis,
    RabiScanResult,
    ReturnData,
    fit_duration_rates,
    plot_duration_rates,
)


@dataclass(frozen=True)
class ConstructivePhaseAnalysis(RabiAnalysis):
    """Maximum-rate relative phase in degrees or an explicit unresolved reason.

    interference_quality describes the ordinary least-squares rate² fit at the
    valid phase coordinates. relative_phase_deg is atan2(C,B), not an IQ phase
    quadrature or logical local-Z correction. No module is written by analysis.
    """

    relative_phase_deg: float | None
    reason: str | None
    interference_quality: FitQuality | None


def analyze_constructive_phase(
    data: RabiScanResult, returns: ReturnData, *, plots: Plots
) -> ConstructivePhaseAnalysis:
    """Fit complete duration traces, then select the rate² interference maximum.

    Exclude unresolved slices below 1/actual duration span. More than three valid
    phases and a full-rank sine/cosine design are needed for finite uncertainty.
    Reject numerical degeneracy/nonfinite uncertainty, not low contrast/sigma or
    residual quality. Nyquist sampling is the operator's responsibility.
    """
    rates = fit_duration_rates(data, returns)
    plot_duration_rates(
        data, rates, coordinate_label="F2 relative to F1 phase (deg)", plots=plots
    )
    base = RabiAnalysis(returns, rates)
    phases = np.array(
        [
            coordinate
            for coordinate, rate in zip(data.coordinates, rates, strict=True)
            if rate.rate is not None
        ]
    )
    squared = np.array([rate.rate**2 for rate in rates if rate.rate is not None])
    if len(phases) <= 3:
        return ConstructivePhaseAnalysis(
            base.return_data,
            rates,
            None,
            "Need more than three identifiable phase slices",
            None,
        )
    radians = np.deg2rad(phases)
    design = np.column_stack((np.ones_like(radians), np.cos(radians), np.sin(radians)))
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, squared, rcond=None)
        if rank != 3:
            return ConstructivePhaseAnalysis(
                returns,
                rates,
                None,
                "Degenerate phase coverage: sine/cosine design is rank deficient",
                None,
            )
        fitted = design @ coefficients
        covariance = (
            np.linalg.inv(design.T @ design)
            * np.sum((squared - fitted) ** 2)
            / (len(phases) - 3)
        )
    except np.linalg.LinAlgError as exc:
        return ConstructivePhaseAnalysis(
            returns, rates, None, f"Interference fit failed: {exc}", None
        )
    quality = compute_fit_quality(
        squared,
        fitted,
        dict(zip(("A", "B", "C"), coefficients, strict=True)),
        covariance,
    )
    a, b, c = coefficients
    modulation = np.hypot(b, c)
    # Only reject a zero at floating-point roundoff, not a physical contrast gate.
    numerical_zero = 8 * np.finfo(float).eps * np.linalg.norm(coefficients)
    if modulation <= numerical_zero:
        return ConstructivePhaseAnalysis(
            returns,
            rates,
            None,
            "Zero rate² modulation: constructive phase undefined",
            quality,
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
        return ConstructivePhaseAnalysis(
            returns,
            rates,
            None,
            "Nonfinite interference fit or unavailable phase uncertainty",
            quality,
        )
    phase = float(np.rad2deg(np.arctan2(c, b)))
    _, ax = plots.subplots("constructive_phase", figsize=(9, 6), layout="constrained")
    dense = np.linspace(-180.0, 180.0, 721)
    ax.plot(phases, squared, "o", label="Identifiable duration-slice rates²")
    ax.plot(
        dense,
        a + b * np.cos(np.deg2rad(dense)) + c * np.sin(np.deg2rad(dense)),
        label="A + B cosφ + C sinφ",
    )
    ax.axvline(phase, ls="--", label=f"Maximum rate at {phase:.4g}°")
    ax.set(
        xlabel="F2 relative to F1 phase (deg)",
        ylabel="Rabi rate² (MHz²)",
        title="Constructive phase · fixed signed gain ratio · model-dependent",
    )
    ax.legend()
    return ConstructivePhaseAnalysis(returns, rates, phase, None, quality)
