"""Fitting algorithms for flux-dependent analysis.

This module provides functions for fitting flux-dependent spectroscopy data,
including candidate breakpoint search, database search, and spectrum fitting.
"""

from __future__ import annotations

import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.figure import Figure
from numba import set_num_threads
from numpy.typing import NDArray
from scipy.optimize import least_squares
from tqdm.auto import tqdm
from typing_extensions import Literal, Optional, overload

from zcu_tools.notebook.persistance import TransitionDict
from zcu_tools.progress_bar import make_pbar
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux

from .models import compile_transitions, count_max_evals, energy2linearform


@lru_cache(maxsize=4)
def _load_database_cached(
    datapath: str, _mtime: float, _size: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load (fluxs, params, energies) from a fluxonium database, cached on the file.

    The database file (~290 MB for the full grid) is re-read on every search; the
    GUI re-runs the search against the *same* database many times (each parameter
    tweak), so reading it once and serving the cached arrays cuts ~0.13 s off every
    repeat call. The cache key includes the file's mtime and size, so editing or
    regenerating the database invalidates the entry rather than serving stale data.
    The returned arrays are the cache's own copies — callers must NOT mutate them.
    """
    with File(datapath, "r") as file:
        if "fluxs" in file:
            f_fluxs = file["fluxs"][:]  # (P,) # type: ignore[index]
        elif "flxs" in file:  # legacy typo
            f_fluxs = file["flxs"][:]  # (P,) # type: ignore[index]
        else:
            raise KeyError("Database file must contain 'fluxs' or 'flxs' dataset.")
        f_params = file["params"][:]  # (N, 3) # type: ignore[index]
        f_energies = file["energies"][:]  # (N, P, M) # type: ignore[index]
    assert isinstance(f_fluxs, np.ndarray)
    assert isinstance(f_params, np.ndarray)
    assert isinstance(f_energies, np.ndarray)
    return (
        np.ascontiguousarray(f_fluxs, dtype=np.float64),
        np.ascontiguousarray(f_params, dtype=np.float64),
        np.ascontiguousarray(f_energies, dtype=np.float64),
    )


def load_database(
    datapath: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Load (f_fluxs, f_params, f_energies) from a fluxonium database, file-cached.

    Stats the file for its (mtime, size) and serves a cached load when unchanged
    (see ``_load_database_cached``). Returns float64 C-contiguous arrays shared with
    the cache — treat them as read-only.
    """
    stat = os.stat(datapath)
    return _load_database_cached(datapath, stat.st_mtime, stat.st_size)


@overload
def search_in_database(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    datapath: str,
    transitions: TransitionDict,
    EJb: tuple[float, float],
    ECb: tuple[float, float],
    ELb: tuple[float, float],
    *,
    n_jobs: int = 1,
    plot: Literal[True] = True,
) -> tuple[tuple[float, float, float], Figure]: ...


@overload
def search_in_database(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    datapath: str,
    transitions: TransitionDict,
    EJb: tuple[float, float],
    ECb: tuple[float, float],
    ELb: tuple[float, float],
    *,
    n_jobs: int = 1,
    plot: Literal[False],
) -> tuple[tuple[float, float, float], None]: ...


def search_in_database(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    datapath: str,
    transitions: TransitionDict,
    EJb: tuple[float, float],
    ECb: tuple[float, float],
    ELb: tuple[float, float],
    *,
    n_jobs: int = 1,
    plot: bool = True,
) -> tuple[tuple[float, float, float], Optional[Figure]]:
    """Search a precomputed fluxonium database for (EJ, EC, EL) best matching
    the observed (fluxs, freqs).

    For each database entry `f_params[i]`, we find a scalar scale `a` (via
    exact breakpoint search on the transition model |a*B + C|) that minimises the
    mean distance to the observed frequencies; the candidate parameters are
    `f_params[i] * a`. The best scale across all entries defines the final
    fit. Scales allow a single simulated grid to cover a continuous (EJ,EC,EL)
    neighbourhood — they are not arbitrary fit degrees of freedom.

    The search is exact but does not scan every entry. A cheap parallel pass
    computes a true lower bound LB(entry) <= min_a F(a) per entry; entries are then
    searched (exactly) in increasing-LB order while tracking the incumbent best
    distance, stopping once LB > incumbent — every remaining entry provably cannot
    win, so the result is identical to a full scan, but typically only a tiny
    fraction of entries are searched (``n_jobs`` sets the LB pass's numba threads).
    """
    from .njit import (
        _apply_interp,
        _interp_weights,
        _lower_bound_kernel,
        search_one_entry,
    )

    # Load the database (file-cached: the GUI re-runs the search against the same
    # database on every parameter tweak, so the ~290 MB read is amortised).
    f_fluxs, f_params, f_energies = load_database(datapath)
    M = f_energies.shape[2]

    # Pre-compile transitions once so the hot loop calls only nogil njit code.
    tr_pairs, tr_coeffs, tr_offsets = compile_transitions(transitions, M)

    # Only the energy levels actually referenced by the transitions enter the
    # linear form, so interpolate (and carry into the parallel kernel) just those
    # levels instead of all M. This shrinks the interpolated array — for the usual
    # 0->1/0->2/1->2 set that is 3 of 15 levels, a ~5x smaller working set — which
    # both speeds the interpolation and lets the bandwidth-bound parallel kernel
    # scale better. The transition pairs are remapped to the reduced level index
    # space so the linear form is numerically identical.
    used_levels = np.unique(tr_pairs.reshape(-1)) if tr_pairs.size else np.arange(M)
    level_pos = np.full(M, -1, dtype=np.int64)
    level_pos[used_levels] = np.arange(used_levels.shape[0])
    tr_pairs_reduced = level_pos[tr_pairs].astype(np.int32)

    # Interpolate points. f_fluxs is strictly increasing and shared across all
    # entries, so precompute (idx, w) once then apply with a parallel njit
    # kernel — avoids N*M Python-level np.interp calls.
    fluxs = np.mod(fluxs, 1.0)
    fluxs_c = np.ascontiguousarray(fluxs, dtype=np.float64)
    idxs, ws = _interp_weights(fluxs_c, f_fluxs)
    energies_used = np.ascontiguousarray(f_energies[:, :, used_levels])
    sf_energies = _apply_interp(energies_used, idxs, ws)

    # Initialize variables
    N = f_params.shape[0]
    best_idx = 0
    best_scale = 1.0
    best_dist = np.inf
    best_params = np.full(3, np.nan)
    # results[i] = (mean distance, scale) per entry. The exact path fills only the
    # entries it actually searches (the prune skips provably-worse ones); the rest
    # keep their lower bound (a valid distance floor) for the diagnostic scatter.
    results = np.full((N, 2), np.nan)  # (N, 2)

    def find_close_points(freqs, energies, scale, allows) -> np.ndarray:
        Bs, Cs = energy2linearform(energies, allows)
        fs = np.abs(scale * Bs + Cs)
        dists = np.abs(fs - freqs[:, None])
        min_idx = np.argmin(dists, axis=1)
        return fs[range(len(freqs)), min_idx]

    # Ensure contiguous float64 for njit signature.
    sf_energies_c = np.ascontiguousarray(sf_energies, dtype=np.float64)
    f_params_c = np.ascontiguousarray(f_params, dtype=np.float64)
    freqs_c = np.ascontiguousarray(freqs, dtype=np.float64)

    n_workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    set_num_threads(n_workers)

    # Exact search with a lower-bound prune. The objective per entry is
    # F(a) = mean_i min_j |A_i - |a*B_ij + C_ij||; ``entry_lower_bound`` gives a
    # valid floor LB(entry) <= min_a F(a) in O(N*K²). Searching entries in
    # increasing-LB order while tracking the incumbent best distance lets us STOP
    # once LB > incumbent — every remaining entry provably cannot beat it, so the
    # winner is IDENTICAL to scanning all entries, but typically only a tiny
    # fraction are fully searched (the true match sorts to the front and drives
    # the incumbent to ~0). The parallel LB pass is cheap; the exact per-entry
    # ``candidate_breakpoint_search`` is the part we avoid for pruned entries.
    lbs = _lower_bound_kernel(
        sf_energies_c,
        f_params_c,
        tr_pairs_reduced,
        tr_coeffs,
        tr_offsets,
        freqs_c,
        EJb[0],
        EJb[1],
        ECb[0],
        ECb[1],
        ELb[0],
        ELb[1],
    )
    results[:, 0] = lbs  # unsearched entries keep their LB for the scatter
    order = np.argsort(lbs)
    idx_bar = make_pbar(total=N, desc="Searching...")
    searched = 0
    try:
        for oi in order:
            oi = int(oi)
            lb = lbs[oi]
            if not np.isfinite(lb) or lb > best_dist:
                break  # all remaining entries have LB >= this -> cannot win
            p0, p1, p2 = f_params[oi]
            a_min = max(EJb[0] / p0, ECb[0] / p1, ELb[0] / p2)
            a_max = min(EJb[1] / p0, ECb[1] / p1, ELb[1] / p2)
            d, a = search_one_entry(
                sf_energies_c[oi],
                tr_pairs_reduced,
                tr_coeffs,
                tr_offsets,
                freqs_c,
                a_min,
                a_max,
            )
            results[oi] = d, a
            searched += 1
            if searched % 64 == 0:
                idx_bar.update(64)
            if d < best_dist:
                best_dist = d
                best_scale = a
                best_idx = oi
                best_params = f_params[oi] * a
        idx_bar.set_description("Done! ")
    except KeyboardInterrupt:
        pass
    finally:
        idx_bar.close()

    if not np.isfinite(best_dist):
        raise RuntimeError(
            "No valid candidate found in database (all parameter bounds infeasible)."
        )

    fig: Figure | None = None
    if plot:
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

        fig.suptitle(
            f"Best Distance: {best_dist:.2g}, EJ={best_params[0]:.2f}, EC={best_params[1]:.2f}, EL={best_params[2]:.2f}"
        )

        # The kernel ran on the reduced level set; reconstruct the best entry's
        # full-level interpolated energies for the (full transition dict) plot.
        best_full = _apply_interp(
            np.ascontiguousarray(f_energies[best_idx : best_idx + 1]), idxs, ws
        )[0]
        p_freqs = find_close_points(freqs, best_full, best_scale, transitions)

        # Frequency comparison plot
        ax_freq = fig.add_subplot(gs[:, 0])
        ax_freq.scatter(fluxs, freqs, label="Target", color="blue", marker="o")
        ax_freq.scatter(fluxs, p_freqs, label="Predicted", color="red", marker="x")
        ax_freq.set_ylabel("Frequency (GHz)")
        ax_freq.set_xlabel("Flux")
        ax_freq.legend()
        ax_freq.grid(True)

        # Per-parameter distance scatter (4k points each — rasterize).
        dists, scales = results[:, 0], results[:, 1]
        finite_dists = dists[np.isfinite(dists)]
        y_top = float(np.max(finite_dists) * 1.1) if finite_dists.size else None
        for i, (name, bound) in enumerate([("EJ", EJb), ("EC", ECb), ("EL", ELb)]):
            ax_param = fig.add_subplot(gs[i, 1])
            ax_param.set_xlim(*bound)
            ax_param.set_xlabel(name)
            ax_param.set_ylabel("Distance")
            ax_param.grid()

            ax_param.scatter(f_params[:, i] * scales, dists, s=2, rasterized=True)
            ax_param.scatter(
                [best_params[i]], [best_dist], color="red", s=50, marker="*"
            )
            if y_top is not None:
                ax_param.set_ylim(0.0, y_top)

        plt.show()

    return (float(best_params[0]), float(best_params[1]), float(best_params[2])), fig


def fit_spectrum(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    init_params: tuple[float, float, float],
    transitions: TransitionDict,
    param_b: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    maxfun: int = 1000,
) -> tuple[float, float, float]:
    max_lvl = count_max_evals(transitions)

    pbar = tqdm(desc="Distance: nan", total=maxfun, leave=False)

    def update_pbar(params, dist) -> None:
        nonlocal pbar

        pbar.set_postfix_str(f"({params[0]:.3f}, {params[1]:.3f}, {params[2]:.2f})")
        pbar.set_description_str(f"Distance: {dist:.2g}")
        pbar.update()

    # 使用 least_squares 進行參數最佳化
    def residuals(params) -> np.ndarray:
        nonlocal fluxs, transitions, freqs

        # 計算能量並轉成線性形式
        _, energies = calculate_energy_vs_flux(
            params, fluxs, cutoff=45, evals_count=max_lvl
        )
        Bs, Cs = energy2linearform(energies, transitions)
        # 計算每個點的最小誤差
        dists = np.min(np.abs(freqs[:, None] - np.abs(Bs + Cs)), axis=1)

        update_pbar(params, np.mean(dists))

        return dists

    import scqubits.settings as scq_settings

    old = scq_settings.PROGRESSBAR_DISABLED
    scq_settings.PROGRESSBAR_DISABLED = True

    EJb, ECb, ELb = param_b
    try:
        res = least_squares(
            residuals,
            init_params,
            bounds=((EJb[0], ECb[0], ELb[0]), (EJb[1], ECb[1], ELb[1])),
            max_nfev=maxfun,
            loss="soft_l1",
        )
    finally:
        scq_settings.PROGRESSBAR_DISABLED = old
        pbar.close()

    if isinstance(res, np.ndarray):  # old version
        best_params = res
    else:
        best_params = res.x

    return (float(best_params[0]), float(best_params[1]), float(best_params[2]))
