"""Fitting algorithms for flux-dependent analysis.

This module provides functions for fitting flux-dependent spectroscopy data,
including candidate breakpoint search, database search, and spectrum fitting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from matplotlib.figure import Figure
from numba import set_num_threads
from numpy.typing import NDArray
from scipy.optimize import least_squares
from tqdm.auto import tqdm, trange
from typing_extensions import Literal, Optional, overload

from zcu_tools.notebook.persistance import TransitionDict
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux

from .models import compile_transitions, count_max_evals, energy2linearform


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
    fuzzy: bool = True,
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
    fuzzy: bool = True,
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
    fuzzy: bool = True,
    plot: bool = True,
) -> tuple[tuple[float, float, float], Optional[Figure]]:
    """Search a precomputed fluxonium database for (EJ, EC, EL) best matching
    the observed (fluxs, freqs).

    For each database entry `f_params[i]`, we find a scalar scale `a` (via
    breakpoint search on the transition model |a*B + C|) that minimises the
    mean distance to the observed frequencies; the candidate parameters are
    `f_params[i] * a`. The best scale across all entries defines the final
    fit. Scales allow a single simulated grid to cover a continuous (EJ,EC,EL)
    neighbourhood — they are not arbitrary fit degrees of freedom.

    The entire outer loop runs inside a parallel njit kernel (``prange``)
    with ``n_jobs`` numba threads (``-1`` = all cores, ``1`` = serial), so no
    Python code sits on the hot path. The kernel is called in batches purely
    to give tqdm progress feedback.
    """
    from .njit import _apply_interp, _interp_weights, _search_kernel

    # Load data from database
    with File(datapath, "r") as file:
        f_fluxs = file["fluxs"][:]  # (f_fluxs, ) # type: ignore[index]
        f_params = file["params"][:]  # (N, 3) # type: ignore[index]
        f_energies = file["energies"][:]  # (N, f_fluxs, M) # type: ignore[index]
    assert isinstance(f_fluxs, np.ndarray)
    assert isinstance(f_params, np.ndarray)
    assert isinstance(f_energies, np.ndarray)

    # Interpolate points. f_fluxs is strictly increasing and shared across all
    # entries, so precompute (idx, w) once then apply with a parallel njit
    # kernel — avoids 4396*M Python-level np.interp calls.
    fluxs = np.mod(fluxs, 1.0)
    f_energies_c = np.ascontiguousarray(f_energies, dtype=np.float64)
    f_fluxs_c = np.ascontiguousarray(f_fluxs, dtype=np.float64)
    fluxs_c = np.ascontiguousarray(fluxs, dtype=np.float64)
    idxs, ws = _interp_weights(fluxs_c, f_fluxs_c)
    sf_energies = _apply_interp(f_energies_c, idxs, ws)

    # Initialize variables
    N = f_params.shape[0]
    best_idx = 0
    best_scale = 1.0
    best_dist = np.inf
    best_params = np.full(3, np.nan)
    results = np.full((N, 2), np.nan)  # (N, 2)

    idx_bar = trange(N, desc="Searching...")

    # Pre-compile transitions once so the hot loop calls only nogil njit code.
    tr_pairs, tr_coeffs, tr_offsets = compile_transitions(
        transitions, f_energies.shape[2]
    )

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

    import os

    n_workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    set_num_threads(n_workers)

    def _run_kernel(start: int, end: int, fuzzy_flag: bool) -> NDArray[np.float64]:
        return _search_kernel(
            sf_energies_c[start:end],
            f_params_c[start:end],
            tr_pairs,
            tr_coeffs,
            tr_offsets,
            freqs_c,
            EJb[0],
            EJb[1],
            ECb[0],
            ECb[1],
            ELb[0],
            ELb[1],
            fuzzy_flag,
        )

    # ~20 batches amortise prange dispatch (sweet spot 256–1024 on this
    # workload) while still giving the pbar enough ticks to feel live. Floor
    # at 64 so small-N cases still hit multiple threads per batch.
    batch_size = max(64, (N + 19) // 20)

    try:
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            results[start:end] = _run_kernel(start, end, fuzzy)
            idx_bar.update(end - start)

        finite_mask = np.isfinite(results[:, 0])
        if finite_mask.any():
            best_idx = int(np.argmin(np.where(finite_mask, results[:, 0], np.inf)))
            best_dist = float(results[best_idx, 0])
            best_scale = float(results[best_idx, 1])
            best_params = f_params[best_idx] * best_scale

        idx_bar.set_description_str("Done! ")
        if fuzzy and np.isfinite(best_dist):
            # recalculate scale with exact method and keep results consistent
            single = _run_kernel(best_idx, best_idx + 1, False)
            best_dist = float(single[0, 0])
            best_scale = float(single[0, 1])
            best_params = f_params[best_idx] * best_scale
            results[best_idx] = best_dist, best_scale

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

        p_freqs = find_close_points(
            freqs, sf_energies[best_idx], best_scale, transitions
        )

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
