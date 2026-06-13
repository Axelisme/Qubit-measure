"""Data processing functions for flux-dependent analysis.

This module provides functions for processing flux-dependent spectroscopy data,
including removing close points, preprocessing data, and analyzing spectra.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def cast2real_and_norm(
    signals: NDArray, use_phase: bool = True, sigma: float = 1
) -> NDArray[np.float64]:
    """Convert complex signals to real with maximum snr"""

    if use_phase:
        signals = signals - np.ma.mean(signals, axis=1, keepdims=True)
        signals = gaussian_filter1d(signals, sigma=sigma, axis=1)
        real_signals = np.abs(signals)
        std = np.ma.std(real_signals, axis=1, keepdims=True)
        # A constant (or fully masked) row has zero/undefined std and carries no
        # spectral feature. Normalising it by 1 keeps it finite and flat so the
        # downstream find_peaks sees no peak, instead of emitting inf/nan from a
        # divide-by-zero that would later poison peak detection.
        std = np.where(np.ma.getdata(std) > 0, std, 1.0)
        real_signals /= std
    else:
        real_signals = np.abs(signals)
        real_signals = gaussian_filter1d(real_signals, sigma=sigma, axis=1)

    return real_signals


def spectrum2d_findpoint(
    dev_values: NDArray[np.float64],
    freqs: NDArray[np.float64],
    real_signals: NDArray[np.float64],
    threshold: float,
    weight: np.ndarray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Find points in a 2D spectrum.
    """

    if weight is not None:
        real_signals = real_signals * weight

    s_mAs = []
    s_freqs = []
    for i in range(real_signals.shape[0]):
        peaks, _ = find_peaks(
            real_signals[i, :], height=threshold, width=(1, None), distance=5
        )

        # If too many peaks, take the top 3
        if len(peaks) > 3:
            peaks = peaks[np.argsort(real_signals[i, peaks])[-3:]]

        s_mAs.extend(dev_values[i] * np.ones(len(peaks)))
        s_freqs.extend(freqs[peaks])
    return np.array(s_mAs), np.array(s_freqs)


def downsample_points(
    xs: NDArray[np.float64], ys: NDArray[np.float64], threshold: float
) -> NDArray[np.bool_]:
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


def diff_mirror(
    xs: NDArray[np.float64], data: NDArray, center: float
) -> NDArray[np.float64]:
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
    c_idx = (len(xs) - 1) * (center - xs[0]) / (xs[-1] - xs[0])

    # 建立正向索引 0,1,2,...,N-1
    idxs = np.arange(data.shape[0])

    # 鏡像後的索引: j = 2*c_idx - i  (四捨五入到最近整數)
    mirror_idxs = np.round(2 * c_idx - idxs).astype(int)

    # 篩掉超出邊界的索引
    valid = (mirror_idxs >= 0) & (mirror_idxs < data.shape[0])

    # 預先建立輸出陣列，以零填充無效位置
    diff_data = np.zeros_like(data, dtype=np.float64)

    # 只在有效位置計算差值。data 可能含 NaN（未量測/遮罩點）；此處刻意讓 NaN
    # 透過 subtract/abs 傳播成 NaN，由下游 loss 計算以 isnan/非零過濾掉，因此
    # 抑制 numpy 對 NaN 運算的 invalid-value 警告（這是預期且已被處理的行為）。
    with np.errstate(invalid="ignore"):
        # 1D 與多維皆靠 broadcasting，一次處理剩餘維度
        diff_data[valid] = np.abs(data[valid] - data[mirror_idxs[valid]])

    return diff_data
