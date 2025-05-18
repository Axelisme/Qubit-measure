"""Data processing functions for flux-dependent analysis.

This module provides functions for processing flux-dependent spectroscopy data,
including removing close points, preprocessing data, and analyzing spectra.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def cast2real_and_norm(signals: np.ndarray, minus_mean: bool = True) -> np.ndarray:
    """
    Convert complex signals to real with maximum snr
    """

    if minus_mean:
        signals = signals - np.ma.mean(signals, axis=0)
    signals = gaussian_filter1d(signals, sigma=1, axis=0)
    amps = np.abs(signals)
    amps /= np.ma.std(amps, axis=0)
    return amps


def spectrum2d_findpoint(
    mAs: np.ndarray,
    fpts: np.ndarray,
    signals: np.ndarray,
    threshold: float,
    weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find points in a 2D spectrum.
    """

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
    """
    Downsample points in a 2D spectrum.
    """

    selected = []
    mask = np.zeros_like(xs, dtype=bool)

    idxs = np.random.Generator(np.random.PCG64(0)).permutation(len(xs))
    for i in idxs:
        if i in selected:
            continue

        too_close = False
        for j in selected:
            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 < threshold**2:
                too_close = True
                break

        if not too_close:
            for j in idxs:
                if xs[i] == xs[j] and j not in selected:
                    selected.append(j)
                    mask[j] = True

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
