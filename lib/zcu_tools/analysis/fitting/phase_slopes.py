"""Shared multi-group quadrature slopes with per-group integer-axis alias checks."""

from dataclasses import dataclass
from math import gcd

import numpy as np
from numpy.typing import NDArray

from .shared import FitTrace, ParameterSpec, SharedFitResult, fit_shared


@dataclass(frozen=True)
class PhaseSlopeFit:
    """Joint quadrature fits for two channels and F coordinate slices.

    slopes_deg: float64 (2,F) canonical slopes in degrees per integer step.
    errors_deg: float64 (2,F) residual-scaled 1σ slope errors in degrees/step.
    fits: channel-major (2,F) tuples of SharedFitResult or None on failure.
        Each result owns slope, amplitude_h and phase_h (degrees), and their
        covariance in that parameter order with product units.
    reasons: channel-major (2,F) tuples, None on success or failure text.
        Unavailable slopes and errors are NaN; other channels/slices continue.
    """

    slopes_deg: NDArray[np.float64]
    errors_deg: NDArray[np.float64]
    fits: tuple[tuple[SharedFitResult | None, ...], ...]
    reasons: tuple[tuple[str | None, ...], ...]

    def accumulated_phase_deg(
        self,
        n: NDArray[np.int64] | NDArray[np.float64],
        *,
        channel: int,
        coordinate: int,
        group: int,
    ) -> NDArray[np.float64] | None:
        """Evaluate the unwrapped fitted angle in degrees at each integer or real n.

        Select a channel and coordinate slice of the shared fit, and an input
        group for its fitted phase offset. Return None for an unavailable fit;
        invalid array indices or missing group parameters raise IndexError or KeyError.
        """
        fit = self.fits[channel][coordinate]
        if fit is None:
            return None
        return (
            np.asarray(n, dtype=np.float64) * fit.values["slope"]
            + fit.values[f"phase_{group}"]
        )


def _gate_count_alias_reason(n: NDArray[np.int64]) -> str | None:
    """Return the slope alias reason for fewer than two n or difference gcd≠1.

    gcd=1 leaves only the unavoidable 2π slope period. Independent group
    offsets require checking each group's effective n separately, not their
    union.
    """
    if len(n) < 2:
        return "At least two effective gate counts are required to identify a slope"
    divisor = gcd(*(int(value - n[0]) for value in n[1:]))
    if divisor != 1:
        return f"Gate-count differences gcd={divisor}: slope alias period 2π/{divisor}"
    return None


def _quadrature_model(
    n: NDArray[np.float64],
    slope: float,
    amplitude: float,
    phase: float,
) -> NDArray[np.float64]:
    angle = np.deg2rad(n * slope + phase)
    return amplitude * np.column_stack((np.cos(angle), np.sin(angle)))


def _slope_seed(
    observations: list[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ],
) -> tuple[float, list[tuple[float, float]]]:
    # For each candidate slope, solve each group's complex amplitude linearly.
    # Search one full 2π period rather than using an unwrap-selected local basin.
    best_cost = np.inf
    best_slope = 0.0
    best_amplitudes: list[tuple[float, float]] = []
    for slope in np.linspace(-180, 180, 721):
        cost = 0.0
        amplitudes: list[tuple[float, float]] = []
        for n, y, errors in observations:
            angle = np.deg2rad(n * slope)
            c, s = np.cos(angle), np.sin(angle)
            design = np.stack(
                (np.column_stack((c, -s)), np.column_stack((s, c))), axis=1
            )
            weighted = (design / errors[..., None]).reshape(-1, 2)
            coefficients = np.linalg.lstsq(weighted, (y / errors).ravel(), rcond=None)[
                0
            ]
            residual = weighted @ coefficients - (y / errors).ravel()
            cost += float(residual @ residual)
            amplitudes.append(
                (
                    float(np.hypot(*coefficients)),
                    float(np.rad2deg(np.arctan2(coefficients[1], coefficients[0]))),
                )
            )
        if cost < best_cost:
            best_cost, best_slope, best_amplitudes = cost, float(slope), amplitudes
    return best_slope, best_amplitudes


def fit_phase_slopes(
    n: NDArray[np.int64],
    quadratures: NDArray[np.float64],
    variances: NDArray[np.float64],
    acquired_shots: NDArray[np.float64],
) -> PhaseSlopeFit:
    """Fit X=A_h cos(n*slope+φ_h), Y=A_h sin(n*slope+φ_h), without DC.

    n is int64 (N,), an integer step axis. The three float64 inputs have
    shape (G,2,2,N,F): group, channel, X/Y quadrature, step, coordinate slice.
    quadratures are dimensionless; variances have squared quadrature units;
    acquired_shots are totals for each independent projection. Each group's
    amplitude/phase is independent; slope is shared within a channel/slice.
    Phases and slopes use degrees. Nonfinite acquired counts block that fit.
    Zero shots, nonfinite quadratures or nonpositive/nonfinite variance remove
    a step; every group must retain >=2 steps with difference gcd=1.
    Slopes use a canonical [-180,180] representative of their 360° period.
    Alias, incomplete group, optimizer failure or invalid covariance return
    NaN slope/error and a reason. Shape/index errors propagate. Inputs are not
    modified; fitting delegates to shared.fit_shared with residual scaling.
    """
    nf = quadratures.shape[-1]
    slopes = np.full((2, nf), np.nan)
    errors = np.full_like(slopes, np.nan)
    all_fits: list[tuple[SharedFitResult | None, ...]] = []
    all_reasons: list[tuple[str | None, ...]] = []
    for state in range(2):
        fits: list[SharedFitResult | None] = []
        reasons: list[str | None] = []
        for f in range(nf):
            result, reason = _fit_frequency(
                n,
                quadratures[:, state, :, :, f],
                variances[:, state, :, :, f],
                acquired_shots[:, state, :, :, f],
            )
            fits.append(result)
            reasons.append(reason)
            if result is not None:
                slopes[state, f] = result.values["slope"]
                errors[state, f] = np.sqrt(result.variance("slope"))
        all_fits.append(tuple(fits))
        all_reasons.append(tuple(reasons))
    return PhaseSlopeFit(slopes, errors, tuple(all_fits), tuple(all_reasons))


def _fit_frequency(
    n: NDArray[np.int64],
    quadratures: NDArray[np.float64],
    variances: NDArray[np.float64],
    acquired: NDArray[np.float64],
) -> tuple[SharedFitResult | None, str | None]:
    observations: list[
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    for group in range(len(quadratures)):
        if not np.all(np.isfinite(acquired[group])):
            return None, f"Group {group}: incomplete acquired gate-count axis"
        valid = np.all(
            np.isfinite(quadratures[group])
            & np.isfinite(variances[group])
            & (variances[group] > 0)
            & (acquired[group] > 0),
            axis=0,
        )
        reason = _gate_count_alias_reason(n[valid])
        if reason is not None:
            return None, f"Group {group}: {reason}"
        observations.append(
            (
                n[valid].astype(np.float64),
                quadratures[group][:, valid].T,
                np.sqrt(variances[group][:, valid].T),
            )
        )
    slope, amplitudes = _slope_seed(observations)
    traces: list[FitTrace] = []
    parameters = [ParameterSpec("slope", slope, limits=(-180, 180))]
    for group, ((x, y, errors), (amplitude, phase)) in enumerate(
        zip(observations, amplitudes, strict=True)
    ):
        names = ("slope", f"amplitude_{group}", f"phase_{group}")
        parameters.extend(
            (
                ParameterSpec(names[1], amplitude, limits=(0, None)),
                ParameterSpec(names[2], phase),
            )
        )
        traces.append(FitTrace(x, y, _quadrature_model, names, errors))
    try:
        result = fit_shared(traces, parameters)
    except (RuntimeError, ValueError) as exc:
        return None, f"Phase-slope optimization failed: {exc}"
    if not result.diagnostics.valid or not np.all(np.isfinite(result.covariance)):
        return None, "Phase-slope optimizer or covariance unavailable"
    return result, None
