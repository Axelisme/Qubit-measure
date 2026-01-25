"""Fitting algorithms for flux-dependent analysis.

This module provides functions for fitting flux-dependent spectroscopy data,
including candidate breakpoint search, database search, and spectrum fitting.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numba import njit
from scipy.optimize import least_squares
from tqdm.auto import tqdm, trange

from zcu_tools.simulate.fluxonium import calculate_energy_vs_flx

from .models import count_max_evals, energy2linearform


def search_in_database(
    flxs, fpts, datapath, allows, EJb, ECb, ELb, n_jobs=-1, fuzzy=True
) -> Tuple[Tuple[float, float, float], Figure]:
    # Load data from database
    with File(datapath, "r") as file:
        f_flxs = file["flxs"][:]  # (f_flxs, ) # type: ignore[index]
        f_params = file["params"][:]  # (N, 3) # type: ignore[index]
        f_energies = file["energies"][:]  # (N, f_flxs, M) # type: ignore[index]
    assert isinstance(f_flxs, np.ndarray)
    assert isinstance(f_params, np.ndarray)
    assert isinstance(f_energies, np.ndarray)

    # Interpolate points
    flxs = np.mod(flxs, 1.0)
    sf_energies = np.empty((f_params.shape[0], len(flxs), f_energies.shape[2]))
    for n in range(f_params.shape[0]):
        for m in range(f_energies.shape[2]):
            sf_energies[n, :, m] = np.interp(flxs, f_flxs, f_energies[n, :, m])

    # Initialize variables
    best_idx = 0
    best_factor = 1.0
    best_dist = np.inf
    best_params = np.full(3, np.nan)
    results = np.full((f_params.shape[0], 2), np.nan)  # (N, 2)

    idx_bar = trange(f_params.shape[0], desc="Searching...")

    def find_close_points(fpts, energies, factor, allows) -> np.ndarray:
        Bs, Cs = energy2linearform(energies, allows)
        fs = np.abs(factor * Bs + Cs)
        dists = np.abs(fs - fpts[:, None])
        min_idx = np.argmin(dists, axis=1)
        return fs[range(len(fpts)), min_idx]

    # ------------------------------------------------------------
    # define the search functions
    # ------------------------------------------------------------

    @njit(
        "float64(float64[:], float64, float64[:,:], float64[:,:])",
        nogil=True,
    )
    def eval_dist(A: np.ndarray, a: float, B: np.ndarray, C: np.ndarray) -> float:
        """
        計算: mean_i(min_j(|A[i] - |a * B[i, j] + C[i, j]||))
        """
        N = A.shape[0]
        K = B.shape[1]

        dist = 0.0
        for i in range(N):
            # min_diff = float("inf") # this will cause error in python 3.8 numba
            min_diff = np.inf
            for j in range(K):
                diff = np.abs(A[i] - np.abs(a * B[i, j] + C[i, j]))
                if diff < min_diff:
                    min_diff = diff
            dist += min_diff

        return dist / N

    @njit(
        "Tuple((float64, float64))(float64[:], float64[:,:], float64[:,:], float64, float64)",
        nogil=True,
    )
    def candidate_breakpoint_search(
        A: np.ndarray, B: np.ndarray, C: np.ndarray, a_min: float, a_max: float
    ) -> Tuple[float, float]:
        """
        使用候選斷點法尋找最佳的 a 值, 使得目標函數最小化
        目標函數: F(a) = mean_i(min_j(|A[i] - |a * B[i, j] + C[i, j]||))
        假設: A 中所有值都是正的

        Parameters:
        A: 目標向量, numpy 陣列, 形狀 (N,), 所有元素均為正數
        B: 候選向量矩陣, numpy 陣列, 形狀 (N, K)
        C: 偏移矩陣, numpy 陣列, 形狀 (N, K)
        a_min: 最小的 a 值
        a_max: 最大的 a 值

        Returns:
        best_distance: 最小的目標函數值, 如果沒有找到則返回 inf
        best_a: 使得目標函數最小的 a 值, 如果沒有找到則返回 1.0
        """
        N = A.shape[0]
        K = B.shape[1]

        # 評估篩選後的 a 值
        # best_distance = float("inf")
        best_distance = np.inf
        best_a = (a_min + a_max) / 2.0

        for i in range(N):
            for j in range(K):
                if B[i, j] == 0:
                    continue

                a1 = (A[i] - C[i, j]) / B[i, j]
                a2 = (-A[i] - C[i, j]) / B[i, j]

                for a in (a1, a2):
                    if a_min <= a <= a_max:
                        dist = eval_dist(A, a, B, C)

                        if dist < best_distance:
                            best_distance = dist
                            best_a = a

        return best_distance, best_a

    @njit(
        "Tuple((float64, float64))(float64[:], float64[:,:], float64[:,:], float64, float64)",
        nogil=True,
    )
    def smart_fuzzy_search(
        A: np.ndarray, B: np.ndarray, C: np.ndarray, a_min: float, a_max: float
    ) -> Tuple[float, float]:
        """
        結合密度估計和有限評估的方法尋找最佳的 a 值
        先通過密度估計找到可能的高密度區域，然後在這些區域中進行有限的評估

        Parameters:
        A: 目標向量, numpy 陣列, 形狀 (N,)
        B: 候選向量矩陣, numpy 陣列, 形狀 (N, K)
        C: 偏移矩陣, numpy 陣列, 形狀 (N, K)
        a_min: 最小的 a 值
        a_max: 最大的 a 值

        Returns:
        best_distance: 最小的目標函數值
        best_a: 使得目標函數最小的 a 值
        """
        N = A.shape[0]
        K = B.shape[1]

        DOWNSAMPLE_THRESHOLD = 1000
        MAX_BIN_USED = 3
        SAMPLE_RATE_IN_BIN = 0.05

        # 收集所有可能的 a 值
        cand_as = []

        for i in range(N):
            for j in range(K):
                if B[i, j] == 0:
                    continue

                a1 = (A[i] - C[i, j]) / B[i, j]
                a2 = (-A[i] - C[i, j]) / B[i, j]

                if a_min <= a1 <= a_max:
                    cand_as.append(a1)
                if a_min <= a2 <= a_max:
                    cand_as.append(a2)

        # 如果候選值太多，降採樣
        if len(cand_as) >= DOWNSAMPLE_THRESHOLD:
            cand_as.sort()

            # 找到高密度區域
            # 1. 先用直方圖方法分析密度
            num_bins = min(100, max(10, len(cand_as) // 10))
            bin_width = (a_max - a_min) / num_bins

            bin_counts = np.zeros(num_bins, dtype=np.int32)
            for i in range(len(cand_as)):
                bin_idx = min(int((cand_as[i] - a_min) / bin_width), num_bins - 1)
                bin_counts[bin_idx] += 1

            # 找到前5多的bin，選取每個bin降採樣100分之1的數量，與中位數，加入test_as
            sample_as = []
            for bin_idx in np.argsort(-bin_counts)[:MAX_BIN_USED]:
                # 如果bin數量為0，跳過
                if bin_counts[bin_idx] == 0:
                    break

                # 計算該bin的起始和結束範圍
                bin_start = a_min + bin_idx * bin_width
                bin_end = bin_start + bin_width

                # 收集該bin內的所有a值
                bin_as = []
                for a in cand_as:
                    if bin_start <= a < bin_end:
                        bin_as.append(a)

                # 如果bin內有值
                if len(bin_as) > 0:
                    # 添加bin的中位數
                    sample_as.append(np.median(np.array(bin_as)))

                    # 降採樣該bin內的值 (取100分之1)
                    step = max(1, int(len(bin_as) * SAMPLE_RATE_IN_BIN))
                    sample_as.extend(bin_as[:step:])
            cand_as = sample_as

        # 5. 評估所有選出的點，找到最佳的
        # best_dist = float("inf")
        best_dist = np.inf
        best_a = (a_min + a_max) / 2.0

        for a in cand_as:
            dist = eval_dist(A, a, B, C)
            if dist < best_dist:
                best_dist = dist
                best_a = a

        return best_dist, best_a

    # ------------------------------------------------------------
    # search function define done
    # ------------------------------------------------------------

    def process_energy(i, fuzzy) -> Tuple[int, float, float]:
        nonlocal f_params, sf_energies, fpts, allows
        assert isinstance(f_params, np.ndarray)

        param = f_params[i]
        a_min = max(EJb[0] / param[0], ECb[0] / param[1], ELb[0] / param[2])
        a_max = min(EJb[1] / param[0], ECb[1] / param[1], ELb[1] / param[2])
        if a_min > a_max:
            return i, np.inf, 1.0

        Bs, Cs = energy2linearform(sf_energies[i], allows)
        if fuzzy:
            return i, *smart_fuzzy_search(fpts, Bs, Cs, a_min, a_max)
        return i, *candidate_breakpoint_search(fpts, Bs, Cs, a_min, a_max)

    try:
        for i, dist, factor in Parallel(  # type: ignore[reportGeneralTypeIssues]
            return_as="generator_unordered", n_jobs=n_jobs, require="sharedmem"
        )(delayed(process_energy)(i, fuzzy) for i in idx_bar):
            results[i] = dist, factor

            if not np.isnan(dist) and dist < best_dist:
                # Update best result
                best_idx = i
                best_factor = factor
                best_params = f_params[i] * factor
                best_dist = dist
        else:
            idx_bar.set_description_str("Done! ")
        if fuzzy:
            # recalculate factor
            best_idx, best_dist, best_factor = process_energy(best_idx, fuzzy=False)
            best_params = f_params[best_idx] * best_factor

    except KeyboardInterrupt:
        pass
    finally:
        idx_bar.close()

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

    fig.suptitle(
        f"Best Distance: {best_dist:.2g}, EJ={best_params[0]:.2f}, EC={best_params[1]:.2f}, EL={best_params[2]:.2f}"
    )

    p_fpts = find_close_points(fpts, sf_energies[best_idx], best_factor, allows)

    # Frequency comparison plot
    ax_freq = fig.add_subplot(gs[:, 0])
    ax_freq.scatter(flxs, fpts, label="Target", color="blue", marker="o")
    ax_freq.scatter(flxs, p_fpts, label="Predicted", color="red", marker="x")
    ax_freq.set_ylabel("Frequency (GHz)")
    ax_freq.set_xlabel("Flux")
    ax_freq.legend()
    ax_freq.grid(True)

    # Create scatter plots for EJ, EC,
    dists, factors = results[:, 0], results[:, 1]
    for i, (name, bound) in enumerate([("EJ", EJb), ("EC", ECb), ("EL", ELb)]):
        ax_param = fig.add_subplot(gs[i, 1])
        ax_param.set_xlim(*bound)
        ax_param.set_xlabel(name)
        ax_param.set_ylabel("Distance")
        ax_param.grid()

        ax_param.scatter(f_params[:, i] * factors, dists, s=2)
        ax_param.scatter([best_params[i]], [best_dist], color="red", s=50, marker="*")
        ax_param.set_ylim(0.0, np.nanmax(dists[np.isfinite(dists)]) * 1.1)

    plt.show()

    return tuple(best_params), fig


def fit_spectrum(
    flxs, fpts, init_params, allows, param_b, maxfun=1000
) -> Tuple[float, float, float]:
    evals_count = count_max_evals(allows)

    pbar = tqdm(desc="Distance: nan", total=maxfun)

    def update_pbar(params, dist) -> None:
        nonlocal pbar

        pbar.set_postfix_str(f"({params[0]:.3f}, {params[1]:.3f}, {params[2]:.2f})")
        pbar.set_description_str(f"Distance: {dist:.2g}")
        pbar.update()

    # 使用 least_squares 進行參數最佳化
    def residuals(params) -> np.ndarray:
        nonlocal flxs, allows, fpts

        # 計算能量並轉成線性形式
        _, energies = calculate_energy_vs_flx(
            params, flxs, cutoff=45, evals_count=evals_count
        )
        Bs, Cs = energy2linearform(energies, allows)
        # 計算每個點的最小誤差
        dists = np.min(np.abs(fpts[:, None] - np.abs(Bs + Cs)), axis=1)

        update_pbar(params, np.mean(dists))

        return dists

    import scqubits.settings as scq_settings

    scq_settings.PROGRESSBAR_DISABLED, old = True, scq_settings.PROGRESSBAR_DISABLED

    EJb, ECb, ELb = param_b
    res = least_squares(
        residuals,
        init_params,
        bounds=((EJb[0], ECb[0], ELb[0]), (EJb[1], ECb[1], ELb[1])),
        max_nfev=maxfun,
        loss="soft_l1",
    )

    pbar.close()

    scq_settings.PROGRESSBAR_DISABLED = old

    if isinstance(res, np.ndarray):  # old version
        best_params = res
    else:
        best_params = res.x

    return tuple(best_params)
