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
            peaks = peaks[np.argsort(amps[peaks, i])[-3:]]

        s_mAs.extend(mAs[i] * np.ones(len(peaks)))
        s_fpts.extend(fpts[peaks])
    return np.array(s_mAs), np.array(s_fpts)


def downsample_points(xs: np.ndarray, ys: np.ndarray, threshold: float) -> np.ndarray:
    selected = []
    mask = np.zeros_like(xs, dtype=bool)

    # np.random.seed(0)
    idxs = np.random.Generator(np.random.PCG64(0)).permutation(len(xs))
    for i in idxs:
        too_close = False
        for j in selected:
            # allow same x
            if xs[i] == xs[j]:
                continue

            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < threshold**2:
                too_close = True
                break

        if not too_close:
            selected.append(i)
            mask[i] = True

    return mask


def diff_mirror(xs: np.ndarray, data: np.ndarray, center: float) -> np.ndarray:
    """
    計算 data 對於 center 位置的鏡像反轉與原本的差值

    參數:
        xs: 1D 座標陣列，形狀為 (N,)
        data: 數據陣列，形狀為 (N, ...)
        center: 鏡像中心的座標值

    返回:
        差值陣列，形狀與 data 相同
    """
    # 找到最接近 center 的索引
    c_idx = (len(xs) - 1) * (center - xs.min()) / (xs.max() - xs.min())

    diff_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        j = int(2 * c_idx - i + 0.5)
        if 0 <= j < data.shape[0]:
            diff_data[i] = np.abs(data[i] - data[j])

    return diff_data
