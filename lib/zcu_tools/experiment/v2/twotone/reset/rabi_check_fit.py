"""Descriptive fits of averaged reset-check traces, without population calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize_scalar


@dataclass(frozen=True)
class RabiCheckTraceFit:
    """Amplitudes are half peak-to-peak; phase uses A*cos(2*pi*f*gain + phase).

    Phase is absent when the fundamental amplitude is below three times its
    conditional coefficient-noise scale. This is a diagnostic, not a confidence
    interval: the frequency is estimated from the before trace and then fixed.
    """

    offset: float
    amplitude: float
    phase_deg: float | None
    second_harmonic_amplitude: float
    residual_rms: float
    coefficients: NDArray[np.float64]

    def evaluate(
        self, gains: NDArray[np.float64], frequency: float
    ) -> NDArray[np.float64]:
        return (
            _design(gains, frequency, second_harmonic=self.coefficients.size == 5)
            @ self.coefficients
        )


@dataclass(frozen=True)
class RabiCheckFit:
    """Frequency is in cycles/gain; ratios are relative contrast, not fidelity."""

    frequency: float
    before: RabiCheckTraceFit
    reset: RabiCheckTraceFit
    after: RabiCheckTraceFit
    contrast_ratio: float
    phase_difference_deg: float | None


def _design(
    gains: NDArray[np.float64], frequency: float, *, second_harmonic: bool
) -> NDArray[np.float64]:
    angle = 2 * np.pi * frequency * gains
    columns = [np.ones_like(gains), np.cos(angle), np.sin(angle)]
    if second_harmonic:
        columns.extend([np.cos(2 * angle), np.sin(2 * angle)])
    return np.column_stack(columns)


def _fit_trace(
    gains: NDArray[np.float64],
    values: NDArray[np.float64],
    frequency: float,
    *,
    second_harmonic: bool = False,
) -> RabiCheckTraceFit:
    valid = np.isfinite(values)
    matrix = _design(gains[valid], frequency, second_harmonic=second_harmonic)
    if np.count_nonzero(valid) <= matrix.shape[1]:
        raise ValueError("Too few finite points for reset-check fit and residuals")
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, values[valid], rcond=None)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if rank != matrix.shape[1]:
        raise ValueError("Reset-check harmonics are not resolved by the gain sweep")
    residual = values[valid] - matrix @ coefficients
    variance = float(residual @ residual / (matrix.shape[0] - matrix.shape[1]))
    covariance = variance * np.linalg.inv(matrix.T @ matrix)
    noise_scale = float(np.sqrt(np.max(np.diag(covariance)[1:3])))
    numerical_floor = 100 * np.finfo(float).eps * float(np.max(np.abs(values[valid])))
    amplitude = float(np.hypot(*coefficients[1:3]))
    phase = None
    if amplitude > max(3 * noise_scale, numerical_floor):
        phase = float(np.degrees(np.arctan2(-coefficients[2], coefficients[1])))
    harmonic = float(np.hypot(*coefficients[3:5])) if second_harmonic else 0.0
    return RabiCheckTraceFit(
        offset=float(coefficients[0]),
        amplitude=amplitude,
        phase_deg=phase,
        second_harmonic_amplitude=harmonic,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        coefficients=coefficients,
    )


def _reference_frequency(
    gains: NDArray[np.float64], values: NDArray[np.float64]
) -> float:
    valid = np.isfinite(values)
    xs, ys = gains[valid], values[valid]
    if xs.size < 6:
        raise ValueError("At least six finite before-reset points are required")
    scale = float(np.ptp(ys))
    if scale <= 100 * np.finfo(float).eps * float(np.max(np.abs(ys))):
        raise ValueError("Before-reset trace has no resolvable Rabi oscillation")
    # Normalize x and y for scale-independent frequency search. Linear projection
    # avoids FFT assumptions about uniform spacing or missing acquisition points.
    span = float(np.ptp(xs))
    xs = (xs - xs[0]) / span
    ys = (ys - np.mean(ys)) / scale
    upper = 0.5 / float(np.max(np.diff(xs)))
    if upper <= 0.25:
        raise ValueError("Gain sweep is too sparse to resolve a Rabi oscillation")

    def loss(cycles: float) -> float:
        matrix = _design(xs, cycles, second_harmonic=False)
        coefficients = np.linalg.lstsq(matrix, ys, rcond=None)[0]
        residual = ys - matrix @ coefficients
        return float(residual @ residual)

    grid = np.linspace(0.25, upper, max(33, int(16 * upper) + 1))
    best = int(np.argmin([loss(float(frequency)) for frequency in grid]))
    optimum = cast(
        OptimizeResult,
        minimize_scalar(
            loss,
            bounds=(grid[max(0, best - 1)], grid[min(grid.size - 1, best + 1)]),
            method="bounded",
            options={"xatol": 1e-10},
        ),
    )
    if not optimum.success or not np.isfinite(optimum.fun):
        raise ValueError("Before-reset Rabi frequency fit did not converge")
    if min(optimum.x - grid[0], grid[-1] - optimum.x) < 1e-6:
        raise ValueError(
            "Rabi frequency is at the search boundary; widen or densify the sweep"
        )
    return float(optimum.x / span)


def fit_reset_rabi(
    gains: NDArray[np.float64], signals: NDArray[np.float64]
) -> RabiCheckFit:
    """Fit before/reset at f and after at f + 2f on one shared IQ projection.

    Each branch may contain NaNs from partial acquisition. Axes must be finite
    and unique; infinities and insufficient/undersampled data fail explicitly.
    The reference search covers >1/4 cycle up to its largest-gap Nyquist bound.
    No damping, readout calibration, or reset-channel inference is assumed.
    """
    if gains.ndim != 1 or signals.shape != (3, gains.size):
        raise ValueError("Reset-check signals must have shape (3, number of gains)")
    if gains.size < 6 or not np.all(np.isfinite(gains)):
        raise ValueError("At least six finite gains are required")
    if np.any(np.isinf(signals)):
        raise ValueError("Reset-check signals must not contain infinities")
    order = np.argsort(gains)
    gains, signals = gains[order], signals[:, order]
    if np.any(np.diff(gains) <= 0):
        raise ValueError("Reset-check gains must be unique")
    frequency = _reference_frequency(gains, signals[0])
    after_gains = gains[np.isfinite(signals[2])]
    if after_gains.size < 6:
        raise ValueError("At least six finite after-reset points are required")
    if 2 * frequency * np.max(np.diff(after_gains)) >= 0.5:
        raise ValueError("Gain sweep is too sparse to resolve the second harmonic")
    before = _fit_trace(gains, signals[0], frequency)
    if before.phase_deg is None:
        raise ValueError(
            "Before-reset Rabi amplitude is not resolved above residual noise"
        )
    reset = _fit_trace(gains, signals[1], frequency)
    after = _fit_trace(gains, signals[2], frequency, second_harmonic=True)
    phase_difference = None
    if after.phase_deg is not None:
        phase_difference = (after.phase_deg - before.phase_deg + 180) % 360 - 180
    return RabiCheckFit(
        frequency=frequency,
        before=before,
        reset=reset,
        after=after,
        contrast_ratio=after.amplitude / before.amplitude,
        phase_difference_deg=phase_difference,
    )
