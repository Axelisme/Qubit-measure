"""Zero-centered Gaussian-envelope cosine fitting on numeric delay traces."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .base import FitResult, fit_func, fitdecaycos
from .quality import decay_time_bounds


def _gaussian_fringe(
    xs: NDArray[np.float64], *parameters: float
) -> NDArray[np.float64]:
    offset, amplitude, frequency, phase, lifetime = parameters
    return offset + amplitude * np.exp(-((xs / lifetime) ** 2)) * np.cos(
        2 * np.pi * (frequency * xs + phase / 360)
    )


def fit_gauss_decay_fringe(
    xs: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    fit_params: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> tuple[float, float, float, float, NDArray[np.float64], FitResult]:
    """Fit baseline + A*exp(-(x/T)**2)*cos(2*pi*(f*x + phase/360)).

    xs contains acquired nonnegative delays; real_signals is matching float64
    data. The envelope center is fixed at zero, and T is the 1/e time in x
    units, not sigma. Return T/stderr, frequency/stderr in inverse-x units,
    curve at the original xs and (parameters, covariance). Parameter order is
    offset, amplitude, frequency, phase in degrees, T. Optional fit_params
    overrides exponential-fringe seed values; non-None fixedparams fixes values.
    Free T uses decay_time_bounds(xs); fixed T is unchanged. Phase is unbounded
    during optimization and returned modulo 360. Automatic initialization uses
    the existing exponential-fringe fitter and expects regular delay spacing.
    Invalid grids or parameter lengths raise ValueError. Numeric fitting
    failures raise AnalysisFailed; no caller-specific acceptance occurs.
    """
    if fit_params is not None and len(fit_params) != 5:
        raise ValueError("fit_params must contain five fringe parameters")
    if fixedparams is not None and len(fixedparams) != 5:
        raise ValueError("fixedparams must contain five fringe parameters")
    lower, upper = decay_time_bounds(xs)
    seed, _ = fitdecaycos(xs, real_signals, decay_bounds=(lower, upper))
    initial = list(map(float, seed))
    if fit_params is not None:
        initial = [
            value if value is not None else guess
            for value, guess in zip(fit_params, initial, strict=True)
        ]
    initial[4] = min(max(initial[4], lower), upper)
    parameters, covariance = fit_func(
        xs,
        real_signals,
        _gaussian_fringe,
        initial,
        (
            [-np.inf, -np.inf, 0.0, -np.inf, lower],
            [np.inf, np.inf, np.inf, np.inf, upper],
        ),
        fixedparams,
    )
    parameters[3] %= 360
    return (
        float(parameters[4]),
        float(np.sqrt(covariance[4, 4])),
        float(parameters[2]),
        float(np.sqrt(covariance[2, 2])),
        _gaussian_fringe(xs, *parameters),
        (parameters, covariance),
    )
