from typing import List, Optional, Sequence, cast

import numpy as np

from .base import assign_init_p, fit_func


# sinusoidal function
def cosfunc(x, *p):
    """p = [y0, yscale, freq, phase]"""
    y0, yscale, freq, phase = p
    return y0 + yscale * np.cos(2 * np.pi * (freq * x + phase / 360))


def fitcos(
    xdata,
    ydata,
    fitparams: Optional[Sequence[Optional[float]]] = None,
    fixedparams: Optional[Sequence[Optional[float]]] = None,
):
    """fitparams = [y0, yscale, freq, phase]"""
    if fitparams is None:
        fitparams = [None] * 4
    fitparams = list(fitparams)

    if fixedparams is not None and len(fixedparams) != 4:
        raise ValueError(
            "Fixed parameters must be a list of four elements: [y0, yscale, freq, phase]"
        )

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = (np.max(ydata) + np.min(ydata)) / 2
        yscale = (np.max(ydata) - np.min(ydata)) / 2
        min_freq = 0.125 / (xdata[-1] - xdata[0])
        fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
        freq_mask = fft_freqs >= min_freq
        fft = np.fft.fft(ydata)[freq_mask]
        fft_freqs = fft_freqs[freq_mask]
        max_id = np.argmax(np.abs(fft))
        freq = fft_freqs[max_id]
        phase = np.angle(fft[max_id], deg=True) % 360

        assign_init_p(fitparams, [y0, yscale, freq, phase])
    fitparams = cast(List[float], fitparams)

    # bounds
    yscale = fitparams[1]
    freq = fitparams[2]
    bounds = (
        [-np.inf, -1.1 * np.abs(yscale), 0.2 * freq, -360],
        [np.inf, 1.1 * np.abs(yscale), 5 * freq, 360],
    )

    pOpt, pCov = fit_func(xdata, ydata, cosfunc, fitparams, bounds, fixedparams)
    if pOpt[1] < 0:
        pOpt[1] = -pOpt[1]
        pOpt[3] = pOpt[3] + 180
    pOpt[3] = pOpt[3] % 360  # convert phase to 0-360
    return pOpt, pCov


# damped sinusoidal function
def decaycos(x, *p):
    """p = [y0, yscale, freq, phase, decay_time]"""
    y0, yscale, freq, phase, decay_time = p
    return y0 + yscale * np.cos(2 * np.pi * (freq * x + phase / 360)) * np.exp(
        -x / decay_time
    )


def fitdecaycos(
    xdata,
    ydata,
    fitparams: Optional[Sequence[Optional[float]]] = None,
    fixedparams: Optional[Sequence[Optional[float]]] = None,
):
    """return (y0, yscale, freq, phase, decay_time), (pOpt, pCov)"""
    if fitparams is None:
        fitparams = [None] * 5
    fitparams = list(fitparams)

    if fixedparams is not None and len(fixedparams) != 5:
        raise ValueError(
            "Fixed parameters must be a list of five elements: [y0, yscale, freq, phase, decay_time]"
        )

    # guess initial parameters
    if any([p is None for p in fitparams]):
        y0 = (np.max(ydata) + np.min(ydata)) / 2
        yscale = (np.max(ydata) - np.min(ydata)) / 2
        min_freq = 0.125 / (xdata[-1] - xdata[0])
        fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
        freq_mask = fft_freqs >= min_freq
        fft = np.fft.fft(ydata)[freq_mask]
        fft_freqs = fft_freqs[freq_mask]
        max_id = np.argmax(np.abs(fft))
        freq = fft_freqs[max_id]
        phase = np.angle(fft[max_id], deg=True) % 360
        decay_time = xdata[-1] - xdata[0]

        assign_init_p(fitparams, [y0, yscale, freq, phase, decay_time])
    fitparams = cast(List[float], fitparams)

    # bounds
    yscale = fitparams[1]
    freq = fitparams[2]
    decay_time = fitparams[4]
    bounds = (
        [-np.inf, -1.1 * np.abs(yscale), 0.2 * freq, -360, 0],
        [np.inf, 1.1 * np.abs(yscale), 5 * freq, 360, np.inf],
    )

    pOpt, pCov = fit_func(xdata, ydata, decaycos, fitparams, bounds, fixedparams)
    if pOpt[1] < 0:
        pOpt[1] = -pOpt[1]
        pOpt[3] = pOpt[3] + 180
    pOpt[3] = pOpt[3] % 360  # convert phase to 0-360
    return pOpt, pCov
