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
    "float64(float64[:], float64[:,:], float64[:,:], float64, float64)",
    nogil=True,
)
def entry_lower_bound(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    a_min: float,
    a_max: float,
) -> float:
    """A valid lower bound on ``candidate_breakpoint_search``'s mean distance.

    The objective is F(a) = mean_i min_j |A_i - |a*B_ij + C_ij|| over a single shared
    scale ``a``. Relaxing the shared-``a`` constraint — letting *each point* pick its
    own best ``a`` in [a_min, a_max] — can only lower the sum, so

        LB = mean_i  min_{a in [a_min,a_max]}  min_j |A_i - |a*B_ij + C_ij||

    is a true lower bound: LB <= min_a F(a). For each point, the inner objective is a
    min of "W"-shapes whose only kinks are at the term's zero crossing
    (a = -C_ij/B_ij) and its ±A_i breakpoints (a = (±A_i - C_ij)/B_ij); the minimum
    over the range is at one of those (clipped into range) or a range endpoint. O(N*K²).

    Used to prune entries that cannot beat the running incumbent in the exact search
    (an entry with LB > incumbent provably cannot win), so the prune is exact.
    """
    N = A.shape[0]
    K = B.shape[1]
    total = 0.0
    for i in range(N):
        gmin = np.inf
        # candidate a's where this point's per-term distance can be minimal
        for j in range(K):
            if B[i, j] != 0.0:
                inv = 1.0 / B[i, j]
                for a in (
                    -C[i, j] * inv,
                    (A[i] - C[i, j]) * inv,
                    (-A[i] - C[i, j]) * inv,
                ):
                    aa = a_min if a < a_min else (a_max if a > a_max else a)
                    g = np.inf
                    for jj in range(K):
                        d = np.abs(A[i] - np.abs(aa * B[i, jj] + C[i, jj]))
                        if d < g:
                            g = d
                    if g < gmin:
                        gmin = g
        # the range endpoints (cover the case where no breakpoint lies inside)
        for aa in (a_min, a_max):
            g = np.inf
            for jj in range(K):
                d = np.abs(A[i] - np.abs(aa * B[i, jj] + C[i, jj]))
                if d < g:
                    g = d
            if g < gmin:
                gmin = g
        total += gmin
    return total / N


@njit(
    (
        "float64[:]("
        "float64[:,:,:], float64[:,:], int32[:,:], float64[:], float64[:], float64[:],"
        " float64, float64, float64, float64, float64, float64)"
    ),
    nogil=True,
    parallel=True,
)
def _lower_bound_kernel(
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
) -> NDArray[np.float64]:
    """Per-entry ``entry_lower_bound`` over all database entries (parallel).

    Returns ``inf`` for entries whose parameter bounds make the scale range empty
    (``a_min > a_max``), so they are pruned before any search.
    """
    N = f_params.shape[0]
    out = np.empty(N, dtype=np.float64)
    for i in prange(N):
        p0 = f_params[i, 0]
        p1 = f_params[i, 1]
        p2 = f_params[i, 2]
        a_min = max(EJ_lo / p0, EC_lo / p1, EL_lo / p2)
        a_max = min(EJ_hi / p0, EC_hi / p1, EL_hi / p2)
        if a_min > a_max:
            out[i] = np.inf
            continue
        Bs, Cs = energy2linearform_nb(sf_energies[i], pairs, coeffs, offsets)
        out[i] = entry_lower_bound(freqs, Bs, Cs, a_min, a_max)
    return out


@njit(
    "Tuple((float64, float64))(float64[:,:], int32[:,:], float64[:], float64[:], "
    "float64[:], float64, float64)",
    nogil=True,
)
def search_one_entry(
    sf_energies_i: NDArray[np.float64],
    pairs: NDArray[np.int32],
    coeffs: NDArray[np.float64],
    offsets: NDArray[np.float64],
    freqs: NDArray[np.float64],
    a_min: float,
    a_max: float,
) -> tuple[float, float]:
    """Exact (best_dist, best_a) for one entry — the linear form + breakpoint search.

    The per-entry core of the LB-pruned exact search: builds B/C from the entry's
    interpolated energies, then runs the exact ``candidate_breakpoint_search``.
    """
    Bs, Cs = energy2linearform_nb(sf_energies_i, pairs, coeffs, offsets)
    return candidate_breakpoint_search(freqs, Bs, Cs, a_min, a_max)
