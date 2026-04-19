from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(
    "Tuple((float64[:,:], float64[:,:]))(float64[:,:], int32[:,:], float64[:], float64[:])",
    nogil=True,
)
def energy2linearform_nb(
    energies: NDArray[np.float64],
    pairs: NDArray[np.int32],
    coeffs: NDArray[np.float64],
    offsets: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """njit equivalent of ``energy2linearform`` using pre-compiled arrays from
    ``compile_transitions``. Output shape: (N, K) for both Bs and Cs."""
    N = energies.shape[0]
    K = pairs.shape[0]
    Bs = np.empty((N, K))
    Cs = np.empty((N, K))
    for k in range(K):
        i = pairs[k, 0]
        j = pairs[k, 1]
        c = coeffs[k]
        off = offsets[k]
        for n in range(N):
            Bs[n, k] = c * (energies[n, j] - energies[n, i])
            Cs[n, k] = off
    return Bs, Cs


@njit(
    "float64(float64[:], float64, float64[:,:], float64[:,:], float64)",
    nogil=True,
)
def eval_dist_bounded(
    A: NDArray[np.float64],
    a: float,
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    threshold: float,
) -> float:
    """Same as eval_dist but returns np.inf early once the partial mean is
    guaranteed to exceed threshold. Used for branch-and-bound pruning."""
    N = A.shape[0]
    K = B.shape[1]

    limit = threshold * N
    dist = 0.0
    for i in range(N):
        min_diff = np.inf
        for j in range(K):
            diff = np.abs(A[i] - np.abs(a * B[i, j] + C[i, j]))
            if diff < min_diff:
                min_diff = diff
        dist += min_diff
        if dist > limit:
            return np.inf

    return dist / N


@njit(
    "Tuple((float64, float64))(float64[:], float64[:,:], float64[:,:], float64, float64)",
    nogil=True,
)
def candidate_breakpoint_search(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    a_min: float,
    a_max: float,
) -> tuple[float, float]:
    """
    使用候選斷點法尋找最佳的 a 值, 使得目標函數最小化
    目標函數: F(a) = mean_i(min_j(|A[i] - |a * B[i, j] + C[i, j]||))
    假設: A 中所有值都是正的
    """
    N = A.shape[0]
    K = B.shape[1]

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
                    dist = eval_dist_bounded(A, a, B, C, best_distance)

                    if dist < best_distance:
                        best_distance = dist
                        best_a = a

    return best_distance, best_a


@njit(
    "Tuple((float64, float64))(float64[:], float64[:,:], float64[:,:], float64, float64)",
    nogil=True,
)
def smart_fuzzy_search(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    a_min: float,
    a_max: float,
) -> tuple[float, float]:
    """
    結合密度估計和有限評估的方法尋找最佳的 a 值
    先通過密度估計找到可能的高密度區域，然後在這些區域中進行有限的評估
    """
    N = A.shape[0]
    K = B.shape[1]

    DOWNSAMPLE_THRESHOLD = 1000
    COVERAGE_TARGET = 0.6  # 挑選的 bin 至少覆蓋總 candidate 數的 60%
    MAX_BIN_USED = 10
    MAX_EVAL_BUDGET = 200  # 在 top regions 內最多評估的 candidate 數

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

        # 排序後相鄰去重（numba 不支援 set）
        dedup = [cand_as[0]]
        for k in range(1, len(cand_as)):
            if cand_as[k] != dedup[-1]:
                dedup.append(cand_as[k])
        sorted_cands = np.asarray(dedup)

        # 直方圖分析密度
        num_bins = min(100, max(10, len(sorted_cands) // 10))
        bin_width = (a_max - a_min) / num_bins

        bin_counts = np.zeros(num_bins, dtype=np.int32)
        for i in range(len(sorted_cands)):
            bin_idx = min(int((sorted_cands[i] - a_min) / bin_width), num_bins - 1)
            bin_counts[bin_idx] += 1

        total_count = len(sorted_cands)
        sorted_bin_idx = np.argsort(-bin_counts)

        # 挑選 bins：累積到覆蓋率目標或 MAX_BIN_USED
        selected = np.zeros(num_bins, dtype=np.bool_)
        cum = 0
        picked = 0
        for k in range(num_bins):
            bi = sorted_bin_idx[k]
            if bin_counts[bi] == 0:
                break
            selected[bi] = True
            cum += bin_counts[bi]
            picked += 1
            if picked >= MAX_BIN_USED:
                break
            if cum >= COVERAGE_TARGET * total_count:
                break

        # 合併相鄰 bins 成連通區間，每區間在全部 candidate 中做均勻降採樣
        sample_as = []
        b = 0
        while b < num_bins:
            if not selected[b]:
                b += 1
                continue
            start_b = b
            while b < num_bins and selected[b]:
                b += 1
            end_b = b  # exclusive

            region_start = a_min + start_b * bin_width
            region_end = a_min + end_b * bin_width
            lo = np.searchsorted(sorted_cands, region_start, side="left")
            hi = np.searchsorted(sorted_cands, region_end, side="left")
            region = sorted_cands[lo:hi]
            if len(region) == 0:
                continue

            # 區間中位數是該 peak 的強 candidate
            sample_as.append(np.median(region))
            # 每個區間預算 = 總預算 × 覆蓋比例
            region_budget = max(2, int(MAX_EVAL_BUDGET * len(region) / max(cum, 1)))
            step = max(1, len(region) // region_budget)
            for kk in range(0, len(region), step):
                sample_as.append(region[kk])
        cand_as = sample_as

    # 評估所有選出的點，找到最佳的
    best_dist = np.inf
    best_a = (a_min + a_max) / 2.0

    for a in cand_as:
        dist = eval_dist_bounded(A, a, B, C, best_dist)
        if dist < best_dist:
            best_dist = dist
            best_a = a

    return best_dist, best_a


@njit(
    "Tuple((int64[:], float64[:]))(float64[:], float64[:])",
    nogil=True,
)
def _interp_weights(
    xs: NDArray[np.float64], xp: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """For each x in xs, find (idx, w) such that the linear interpolation is
    (1-w) * fp[idx] + w * fp[idx+1]. Assumes xp is strictly increasing.
    Clamps out-of-range xs to the endpoints (matches np.interp behaviour)."""
    n = xs.shape[0]
    P = xp.shape[0]
    idxs = np.empty(n, dtype=np.int64)
    ws = np.empty(n, dtype=np.float64)
    for k in range(n):
        x = xs[k]
        # binary search for largest idx with xp[idx] <= x
        lo = 0
        hi = P - 1
        if x <= xp[0]:
            idxs[k] = 0
            ws[k] = 0.0
            continue
        if x >= xp[P - 1]:
            idxs[k] = P - 2
            ws[k] = 1.0
            continue
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if xp[mid] <= x:
                lo = mid
            else:
                hi = mid
        idxs[k] = lo
        ws[k] = (x - xp[lo]) / (xp[lo + 1] - xp[lo])
    return idxs, ws


@njit(
    "float64[:,:,:](float64[:,:,:], int64[:], float64[:])",
    nogil=True,
    parallel=True,
)
def _apply_interp(
    f_energies: NDArray[np.float64],
    idxs: NDArray[np.int64],
    ws: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply precomputed interp (idxs, ws) along axis 1 of f_energies.

    f_energies shape: (N, P, M). Output shape: (N, len(idxs), M).
    """
    N = f_energies.shape[0]
    J = idxs.shape[0]
    M = f_energies.shape[2]
    out = np.empty((N, J, M), dtype=np.float64)
    for n in prange(N):
        for j in range(J):
            idx = idxs[j]
            w = ws[j]
            one_minus_w = 1.0 - w
            for m in range(M):
                out[n, j, m] = (
                    one_minus_w * f_energies[n, idx, m] + w * f_energies[n, idx + 1, m]
                )
    return out


@njit(
    (
        "float64[:,:]("
        "float64[:,:,:], float64[:,:], int32[:,:], float64[:], float64[:], float64[:],"
        " float64, float64, float64, float64, float64, float64, boolean)"
    ),
    nogil=True,
    parallel=True,
)
def _search_kernel(
    sf_energies: NDArray[np.float64],
    f_params: NDArray[np.float64],
    pairs: NDArray[np.int32],
    coeffs: NDArray[np.float64],
    offsets: NDArray[np.float64],
    freqs: NDArray[np.float64],
    EJ_lo: float,
    EJ_hi: float,
    EC_lo: float,
    EC_hi: float,
    EL_lo: float,
    EL_hi: float,
    fuzzy: bool,
) -> NDArray[np.float64]:
    N = f_params.shape[0]
    results = np.empty((N, 2), dtype=np.float64)
    for i in prange(N):
        p0 = f_params[i, 0]
        p1 = f_params[i, 1]
        p2 = f_params[i, 2]
        a_min = max(EJ_lo / p0, EC_lo / p1, EL_lo / p2)
        a_max = min(EJ_hi / p0, EC_hi / p1, EL_hi / p2)
        if a_min > a_max:
            results[i, 0] = np.inf
            results[i, 1] = 1.0
            continue
        Bs, Cs = energy2linearform_nb(sf_energies[i], pairs, coeffs, offsets)
        if fuzzy:
            d, a = smart_fuzzy_search(freqs, Bs, Cs, a_min, a_max)
        else:
            d, a = candidate_breakpoint_search(freqs, Bs, Cs, a_min, a_max)
        results[i, 0] = d
        results[i, 1] = a
    return results
