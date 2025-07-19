"""Data processing functions for flux-dependent analysis.

This module provides functions for processing flux-dependent spectroscopy data,
including removing close points, preprocessing data, and analyzing spectra.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def cast2real_and_norm(signals: np.ndarray, use_phase: bool = True) -> np.ndarray:
    """
    Convert complex signals to real with maximum snr
    """

    if use_phase:
        signals = signals - np.ma.mean(signals, axis=0)
        signals = gaussian_filter1d(signals, sigma=1, axis=0)
        amps = np.abs(signals)
        amps /= np.ma.std(amps, axis=0)
    else:
        amps = np.abs(signals)
        amps = gaussian_filter1d(amps, sigma=1, axis=0)

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
    # 使用向量化計算，避免 Python for-loop 以提升效能

    # 以 xs 的端點為 0 與 N-1，計算 center 對應的連續索引位置 (浮點數)
    c_idx = (len(xs) - 1) * (center - xs.min()) / (xs.max() - xs.min())

    # 建立正向索引 0,1,2,...,N-1
    idxs = np.arange(data.shape[0])

    # 鏡像後的索引: j = 2*c_idx - i  (四捨五入到最近整數)
    mirror_idxs = np.round(2 * c_idx - idxs).astype(int)

    # 篩掉超出邊界的索引
    valid = (mirror_idxs >= 0) & (mirror_idxs < data.shape[0])

    # 預先建立輸出陣列，以零填充無效位置
    diff_data = np.zeros_like(data, dtype=np.float64)

    # 只在有效位置計算差值
    if data.ndim == 1:
        diff_data[valid] = np.abs(data[valid] - data[mirror_idxs[valid]])
    else:
        # 利用 broadcasting，一次處理剩餘維度
        diff_data[valid] = np.abs(data[valid] - data[mirror_idxs[valid]])

    return diff_data
