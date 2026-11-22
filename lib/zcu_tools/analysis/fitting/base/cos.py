from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.failure import AnalysisFailed

from .base import FitResult, assign_init_p, fit_func


def _prepare_oscillation_trace(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fitparams: Sequence[float | None],
    fixedparams: Sequence[float | None] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if x.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError(
            "Oscillation coordinates and observations require matching nonempty 1-D shapes"
        )
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Oscillation coordinates and observations must be finite")
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    if any(p is None for p in fitparams):
        steps = np.diff(x)
        if (
            steps.size == 0
            or steps[0] <= 0.0
            or np.any(steps < 0.0)
            or not np.isfinite(x[-1] - x[0])
        ):
            raise ValueError(
                "Oscillation axis must be ordered with finite positive span and first step"
            )
    n_free = (
        len(fitparams) if fixedparams is None else sum(p is None for p in fixedparams)
    )
    if y.size < n_free:
        raise AnalysisFailed("Insufficient observations for the free fit parameters")
    return x, y


def _fft_seed(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    free_amplitude_guess: bool,
    require_frequency_interval: bool,
) -> tuple[list[float], tuple[float, float]]:
    high, low = np.max(y), np.min(y)
    if free_amplitude_guess and high == low:
        raise AnalysisFailed("Oscillation trace has no contrast")
    min_freq = 0.125 / (x[-1] - x[0])
    frequencies = np.fft.fftfreq(len(y), d=x[1] - x[0])
    mask = frequencies >= min_freq
    positive = frequencies[mask]
    if positive.size == 0:
        raise AnalysisFailed("Oscillation trace has no positive seed frequency")
    if require_frequency_interval and positive.size == 1:
        raise AnalysisFailed("Oscillation trace has no free frequency interval")
    fft = np.fft.fft(y)[mask]
    peak = int(np.argmax(np.abs(fft)))
    initial: list[float] = [
        (high + low) / 2,
        (high - low) / 2,
        positive[peak],
        np.angle(fft[peak], deg=True) % 360,
    ]
    return initial, (positive[0], positive[-1])


# sinusoidal function
def cosfunc(x: NDArray[np.float64], *p: float) -> NDArray[np.float64]:
    """p = [y0, yscale, freq (1/x), phase (deg)]"""
    y0, yscale, freq, phase = p
    return y0 + yscale * np.cos(2 * np.pi * (freq * x + phase / 360))


def fitcos(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    fitparams: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
) -> FitResult:
    """Fit [y0, yscale, freq (1/x), phase (deg)] and return parameters/covariance.

    Phase is unbounded during optimization and returned modulo 360 degrees.
    ``fitparams`` supplies initial guesses; non-None ``fixedparams`` entries
    hold those parameters fixed. Automatic initialization expects regularly
    spaced coordinates in ascending or descending order. Invalid input raises
    ValueError; valid traces with no usable seed or too few observations for
    the free parameters raise AnalysisFailed.
    """
    if fitparams is None:
        fitparams = [None] * 4
    fitparams = list(fitparams)
    if len(fitparams) != 4 or (fixedparams is not None and len(fixedparams) != 4):
        raise ValueError(
            "Cosine parameters must have four entries: [y0, yscale, freq, phase]"
        )
    xdata, ydata = _prepare_oscillation_trace(xdata, ydata, fitparams, fixedparams)

    # guess initial parameters
    if any(p is None for p in fitparams):
        initial, _ = _fft_seed(
            xdata,
            ydata,
            free_amplitude_guess=fitparams[1] is None
            and (fixedparams is None or fixedparams[1] is None),
            require_frequency_interval=False,
        )
        assign_init_p(fitparams, initial)
    # tolerated(cast): reportArgumentType: FFT guesses fill four unset slots and preserve supplied values; in-place filling cannot narrow the list type.
    fitparams = cast(list[float], fitparams)

    # bounds
    yscale = fitparams[1]
    freq = fitparams[2]
    bounds = (
        [-np.inf, -1.1 * np.abs(yscale), 0.2 * freq, -np.inf],
        [np.inf, 1.1 * np.abs(yscale), 5 * freq, np.inf],
    )

    pOpt, pCov = fit_func(xdata, ydata, cosfunc, fitparams, bounds, fixedparams)
    if pOpt[1] < 0:
        pOpt[1] = -pOpt[1]
        pOpt[3] = pOpt[3] + 180
    pOpt[3] = pOpt[3] % 360  # convert phase to 0-360
    return pOpt, pCov


# damped sinusoidal function
def decaycos(x: NDArray[np.float64], *p: float) -> NDArray[np.float64]:
    """p = [y0, yscale, freq, phase, decay_time]"""
    y0, yscale, freq, phase, decay_time = p
    return y0 + yscale * np.cos(2 * np.pi * (freq * x + phase / 360)) * np.exp(
        -x / decay_time
    )


def fitdecaycos(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    fitparams: Sequence[float | None] | None = None,
    fixedparams: Sequence[float | None] | None = None,
    *,
    decay_bounds: tuple[float, float] | None = None,
) -> FitResult:
    """Fit [y0, yscale, freq (1/x), phase (deg), decay_time] and covariance.

    Phase is unbounded during optimization and returned modulo 360 degrees.
    Amplitude is bounded to 5 times the absolute initial amplitude (the data
    half-range when not supplied), so a fringe whose envelope decays before
    the first extremum still recovers its t=0 amplitude and decay time.
    ``fitparams`` supplies initial guesses; non-None ``fixedparams`` entries
    hold those parameters fixed. Decay time uses the units of ``xdata``.
    Optional decay_bounds=(lower, upper) constrains free decay time and clips
    its initial guess; None keeps positive unbounded decay time. Fixed values
    are not clipped. Invalid bounds raise ValueError.
    Automatic initialization expects regularly spaced coordinates in ascending
    or descending order. Invalid input raises ValueError; valid traces with
    no usable seed or too few observations for the free parameters raise
    AnalysisFailed.
    """
    if fitparams is None:
        fitparams = [None] * 5
    fitparams = list(fitparams)
    if len(fitparams) != 5 or (fixedparams is not None and len(fixedparams) != 5):
        raise ValueError(
            "Damped cosine parameters must have five entries: [y0, yscale, freq, phase, decay_time]"
        )
    xdata, ydata = _prepare_oscillation_trace(xdata, ydata, fitparams, fixedparams)

    # guess initial parameters
    if any(p is None for p in fitparams):
        initial, freq_bound = _fft_seed(
            xdata,
            ydata,
            free_amplitude_guess=fitparams[1] is None
            and (fixedparams is None or fixedparams[1] is None),
            require_frequency_interval=fixedparams is None or fixedparams[2] is None,
        )
        assign_init_p(fitparams, [*initial, xdata[-1] - xdata[0]])
    else:
        freq = fitparams[2]
        assert freq is not None
        freq_bound = (0.2 * freq, 5 * freq)
    # tolerated(cast): reportArgumentType: FFT/decay guesses fill five unset slots and preserve supplied values; in-place filling cannot narrow the list type.
    fitparams = cast(list[float], fitparams)

    # bounds
    yscale = fitparams[1]
    freq = fitparams[2]
    lower, upper = (0.0, np.inf) if decay_bounds is None else decay_bounds
    if decay_bounds is not None:
        fitparams[4] = min(max(fitparams[4], lower), upper)
    # When the envelope decays before the first fringe extremum, the data
    # half-range underestimates the t=0 amplitude; a cap near it is absorbed by
    # a longer decay time. 5x leaves room for that while still bounding the fit.
    bounds = (
        [-np.inf, -5 * np.abs(yscale), freq_bound[0], -np.inf, lower],
        [np.inf, 5 * np.abs(yscale), freq_bound[1], np.inf, upper],
    )

    pOpt, pCov = fit_func(xdata, ydata, decaycos, fitparams, bounds, fixedparams)
    if pOpt[1] < 0:
        pOpt[1] = -pOpt[1]
        pOpt[3] = pOpt[3] + 180
    pOpt[3] = pOpt[3] % 360  # convert phase to 0-360
    return pOpt, pCov
