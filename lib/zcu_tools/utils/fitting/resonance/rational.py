from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import validate_complex_fit_inputs


@dataclass(frozen=True)
class RationalFitResult:
    freq_center: float
    freq_scale: float
    signal_scale: float
    numerator: tuple[complex, complex]
    denominator: tuple[complex, complex]
    residual_rms: float

    def evaluate(self, freqs: NDArray[np.float64]) -> NDArray[np.complex128]:
        x = (freqs - self.freq_center) / self.freq_scale
        a, b = self.numerator
        c, d = self.denominator
        values = self.signal_scale * (a + b * x) / (c + d * x)
        return np.asarray(values, dtype=np.complex128)


def _frequency_derivative_weights(
    xs: NDArray[np.float64], ys: NDArray[np.complex128]
) -> NDArray[np.float64]:
    order = np.argsort(xs)
    sorted_x = xs[order]
    sorted_y = ys[order]
    gradient = np.gradient(sorted_y, sorted_x)
    unsorted = np.empty_like(gradient)
    unsorted[order] = gradient
    weights = 1.0 / np.sqrt(1.0 + np.abs(unsorted) ** 2)
    return np.asarray(weights / np.max(weights), dtype=np.float64)


def fit_degree_one_rational(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    *,
    max_residual_rms: float = np.inf,
) -> RationalFitResult:
    """Fit ``(a + b*x) / (c + d*x)`` with centered/scaled frequency.

    This is an internal initializer primitive, not a final physical model. It uses
    the actual frequency coordinates for derivative weighting, so nonuniform grids
    do not silently become sample-index weighted.
    """
    span = validate_complex_fit_inputs(freqs, signals)
    signal_scale = float(np.sqrt(np.mean(np.abs(signals) ** 2)))
    if not np.isfinite(signal_scale) or signal_scale <= 0.0:
        raise ValueError("rational initializer requires non-zero signal scale")

    freq_center = 0.5 * float(np.min(freqs) + np.max(freqs))
    xs = np.asarray((freqs - freq_center) / span, dtype=np.float64)
    ys = np.asarray(signals / signal_scale, dtype=np.complex128)
    weights = _frequency_derivative_weights(xs, ys)

    # Fix denominator constant to one; any non-degenerate degree-1 rational model
    # whose denominator is non-zero at the center can be represented this way.
    design = np.column_stack(
        (
            np.ones_like(xs, dtype=np.complex128),
            xs.astype(np.complex128),
            -ys * xs,
        )
    )
    weighted_design = design * weights[:, None]
    weighted_target = ys * weights
    if np.linalg.cond(weighted_design) > 1e12:
        raise ValueError("rational initializer linear system is ill-conditioned")
    solution, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
    if not np.all(np.isfinite(solution)):
        raise ValueError("rational initializer produced non-finite coefficients")

    a, b, d = (complex(value) for value in solution)
    denominator = 1.0 + d * xs
    if np.any(np.abs(denominator) < 1e-9):
        raise ValueError("rational initializer denominator is singular on the grid")
    predicted = (a + b * xs) / denominator
    residual_rms = float(np.sqrt(np.mean(np.abs(predicted - ys) ** 2)))
    if not np.isfinite(residual_rms) or residual_rms > max_residual_rms:
        raise ValueError(
            "rational initializer residual is too large "
            f"({residual_rms:.3g} > {max_residual_rms:.3g})"
        )

    return RationalFitResult(
        freq_center=freq_center,
        freq_scale=span,
        signal_scale=signal_scale,
        numerator=(a, b),
        denominator=(1.0 + 0.0j, d),
        residual_rms=residual_rms,
    )
