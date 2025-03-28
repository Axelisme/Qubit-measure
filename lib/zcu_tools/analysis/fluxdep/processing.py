"""Data processing functions for flux-dependent analysis.

This module provides functions for processing flux-dependent spectroscopy data,
including removing close points, preprocessing data, and analyzing spectra.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def mA2flx(mAs, mA_c, period):
    return (mAs - mA_c) / period + 0.5


def flx2mA(flxs, mA_c, period):
    return (flxs - 0.5) * period + mA_c


def format_rawdata(mAs, fpts, spectrum):
    fpts = fpts / 1e9  # convert to GHz
    mAs = mAs * 1e3  # convert to mA

    if mAs[0] > mAs[-1]:  # Ensure that the fluxes are in increasing
        mAs = mAs[::-1]
        spectrum = spectrum[:, ::-1]
    if fpts[0] > fpts[-1]:  # Ensure that the frequencies are in increasing
        fpts = fpts[::-1]
        spectrum = spectrum[::-1, :]

    return mAs, fpts, spectrum


def cast2real_and_norm(signals):
    signals = signals - np.ma.mean(signals, axis=0)
    signals = gaussian_filter1d(signals, sigma=1, axis=0)
    amps = np.abs(signals)
    amps /= np.ma.std(amps, axis=0)
    return amps


def spectrum_analyze(mAs, fpts, signals, threshold, weight=None):
    amps = cast2real_and_norm(signals)

    if weight is not None:
        amps *= weight

    s_mAs = []
    s_fpts = []
    for i in range(amps.shape[1]):
        peaks, _ = find_peaks(amps[:, i], height=threshold, width=(1, None), distance=5)

        # If too many peaks, take the top 3
        if len(peaks) > 3:
            peaks = np.argsort(amps[peaks, i])[-3:]

        s_mAs.extend(mAs[i] * np.ones(len(peaks)))
        s_fpts.extend(fpts[peaks])
    return np.array(s_mAs), np.array(s_fpts)
