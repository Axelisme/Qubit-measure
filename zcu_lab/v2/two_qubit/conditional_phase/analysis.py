"""Count-weighted phase-slope fits. This module contains no plotting or run policy."""

from dataclasses import dataclass
from math import gcd
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.shared import (
    FitTrace,
    ParameterSpec,
    SharedFitResult,
    fit_shared,
)

from .._phase import QuadraturePhase, count_phase, sigma_variance
from .._sums import sigma
from .crossing import PhaseCrossings, phase_crossings


@dataclass(frozen=True)
class CountQuadratures:
    """Count reductions with spectator × quadrature × n × frequency axes.

    values is target sigma, variances is its count-derived variance, acquired
    is each projection's shot total. Empty counts give NaN values/variances.
    """

    values: NDArray[np.float64]
    variances: NDArray[np.float64]
    acquired: NDArray[np.float64]


def count_quadratures(
    counts: NDArray[np.float64],
    target: Literal["f1", "f2"],
) -> CountQuadratures:
    """Reduce terminal (...,3,3) counts to sigma, variance and shot totals."""
    return CountQuadratures(
        sigma(counts, target),
        sigma_variance(counts, target),
        counts.sum(axis=(-2, -1)),
    )


@dataclass(frozen=True)
class PhaseSlopeFit:
    """Shared slopes for actual spectator g/e, each array shaped (2,Nf).

    errors_deg is the residual-scaled shared-fit standard error. fits and reasons
    are spectator-major tuples of per-frequency optimizer results or reasons.
    Unavailable slopes/errors stay NaN, and never authorize writeback.
    """

    slopes_deg: NDArray[np.float64]
    errors_deg: NDArray[np.float64]
    fits: tuple[tuple[SharedFitResult | None, ...], ...]
    reasons: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True)
class ConditionalPhaseAnalysis:
    """Classified diagnostics and per-gate conditional phase from fitted slopes.

    phases is (2,Nn,Nf), quadratures/acquired_shots are (2,2,Nn,Nf).
    slope_fit uses actual spectator g/e. conditional_deg is slope_e−slope_g,
    wrapped to [-180,180), with Nf entries. crossings contains the per-gate π
    intersections. frequency_mhz is set only for a unique complete crossing.
    """

    phases: tuple[tuple[tuple[QuadraturePhase, ...], ...], ...]
    quadratures: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    gate_counts: tuple[int, ...]
    frequencies_mhz: tuple[float, ...]
    slope_fit: PhaseSlopeFit
    conditional_deg: NDArray[np.float64]
    crossings: PhaseCrossings
    frequency_mhz: float | None
    frequency_reason: str | None


def fitted_crossings(
    frequencies: NDArray[np.float64],
    conditional_deg: NDArray[np.float64],
    fit: PhaseSlopeFit,
) -> PhaseCrossings:
    """Find per-gate π crossings, preserving fit-unavailability reasons.

    Inputs are the increasing frequency axis, its conditional slope difference,
    and the fits that produced it. Any unavailable slope prevents selection.
    """
    crossings = phase_crossings(frequencies, conditional_deg)
    reasons = [reason for state in fit.reasons for reason in state if reason]
    if reasons:
        return PhaseCrossings((), "; ".join(dict.fromkeys(reasons)))
    return crossings


def gate_count_alias_reason(n: NDArray[np.int64]) -> str | None:
    """Return the slope alias reason for fewer than two n or difference gcd≠1.

    Integer n are train lengths. gcd=1 leaves only the unavoidable 2π slope
    period. Offsets must be independent between herald groups, so callers check
    each group's effective n separately, never their union.
    """
    if len(n) < 2:
        return "At least two effective gate counts are required to identify a slope"
    divisor = gcd(*(int(value - n[0]) for value in n[1:]))
    if divisor != 1:
        return f"Gate-count differences gcd={divisor}: slope alias period 2π/{divisor}"
    return None


def phase_points(
    quadratures: NDArray[np.float64],
) -> tuple[tuple[tuple[QuadraturePhase, ...], ...], ...]:
    """Convert (spectator,quadrature,n,frequency) sigma to diagnostic phase points."""
    return tuple(
        tuple(
            tuple(count_phase(float(x), float(y)) for x, y in zip(xs, ys, strict=True))
            for xs, ys in zip(state[0], state[1], strict=True)
        )
        for state in quadratures
    )


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

    Inputs have (group,actual spectator,quadrature,n,frequency) axes. Each group's
    A/φ is independent; slope is shared by actual spectator and frequency.
    Variances are count-derived sigma variances. NaN acquired counts mean an
    incomplete axis and block that fit. Zero acceptance or nonpositive variance
    removes n; each group must still have >=2 effective n with difference gcd=1.
    Slopes use a canonical [-180,180] representative of their 2π period.
    Failed optimization returns NaN and a reason, never a writeback value.
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
        reason = gate_count_alias_reason(n[valid])
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
