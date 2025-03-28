"""Fitting algorithms for flux-dependent analysis.

This module provides functions for fitting flux-dependent spectroscopy data,
including candidate breakpoint search, database search, and spectrum fitting.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from IPython.display import display
from joblib import Parallel, delayed
from numba import njit
from tqdm.auto import tqdm, trange

from zcu_tools.tools import AsyncFunc

from .models import energy2linearform


@njit(
    "Tuple((float64, float64))(float64[:], float64[:,:], float64[:,:], float64, float64)",
    nogil=True,
)
def candidate_breakpoint_search(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, a_min: float, a_max: float
) -> Tuple[float, float]:
    """
    使用候選斷點法尋找最佳的 a 值, 使得目標函數最小化
    目標函數: F(a) = sum_i(min_j(|A[i] - |a * B[i, j] + C[i, j]||))
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

    # 找出最佳的 a 值
    best_distance = float("inf")
    best_a = 1.0

    distances = np.empty(N, dtype=np.float64)
    for i in range(N):
        for j in range(K):
            if B[i, j] == 0:
                continue

            a1 = (A[i] - C[i, j]) / B[i, j]
            a2 = (-A[i] - C[i, j]) / B[i, j]

            for a in (a1, a2):
                if a < a_min or a > a_max:
                    continue

                for i in range(N):
                    min_diff = float("inf")
                    for j in range(K):
                        # 計算距離
                        diff = np.abs(A[i] - np.abs(a * B[i, j] + C[i, j]))
                        if diff < min_diff:
                            min_diff = diff
                    distances[i] = min_diff

                total_distance = np.mean(distances)
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_a = a

    return best_distance, best_a


def search_in_database(flxs, fpts, datapath, allows, EJb, ECb, ELb, n_jobs=-1):
    # Load data from database
    with File(datapath, "r") as file:
        f_flxs = file["flxs"][:]  # (f_flxs, )
        f_params = file["params"][:]  # (N, 3)
        f_energies = file["energies"][:]  # (N, f_flxs, M)

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

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

    # Frequency comparison plot
    ax_freq = fig.add_subplot(gs[:, 0])
    ax_freq.scatter(flxs, fpts, label="Target", color="blue", marker="o")
    pred_scatter = ax_freq.scatter(
        flxs, np.zeros_like(fpts), label="Predicted", color="red", marker="x"
    )

    ax_freq.set_ylabel("Frequency (GHz)")
    ax_freq.legend()
    ax_freq.grid(True)

    # Create scatter plots for EJ, EC,
    param_axs = []
    param_scatters = []
    best_param_scatters = []
    name_bounds = [("EJ", EJb), ("EC", ECb), ("EL", ELb)]
    for i in range(3):
        name, bound = name_bounds[i]
        ax_param = fig.add_subplot(gs[i, 1])
        ax_param.set_xlim(bound[0], bound[1])
        ax_param.set_xlabel(name)
        ax_param.set_ylabel("Distance")
        ax_param.grid()
        scatter = ax_param.scatter(
            range(f_params.shape[0]), np.zeros(f_params.shape[0]), s=2
        )
        best_scatter = ax_param.scatter([0], [0], color="red", s=50, marker="*")
        param_axs.append(ax_param)
        param_scatters.append(scatter)
        best_param_scatters.append(best_scatter)

    dh = display(fig, display_id=True)

    def find_close_points(fpts, energies, factor, allows):
        Bs, Cs = energy2linearform(energies, allows)
        fs = np.abs(factor * Bs + Cs)
        dists = np.abs(fs - fpts[:, None])
        min_idx = np.argmin(dists, axis=1)
        return fs[range(len(fpts)), min_idx]

    prev_draw_idx = -1

    def update_plot(_):
        nonlocal best_dist, best_params, results, prev_draw_idx, best_idx

        # Update best result
        if best_idx != prev_draw_idx:
            p_fpts = find_close_points(fpts, sf_energies[best_idx], best_factor, allows)
            pred_scatter.set_offsets(np.c_[flxs, p_fpts])
            ax_freq.set_ylim(np.min([fpts, p_fpts]), np.max([fpts, p_fpts]))

            fig.suptitle(
                f"Best Distance: {best_dist:.2g}, EJ={best_params[0]:.2f}, EC={best_params[1]:.2f}, EL={best_params[2]:.2f}"
            )
            prev_draw_idx = best_idx

        # Update scatter plots
        dists, factors = results[:, 0], results[:, 1]
        if np.sum(np.isfinite(dists)) > 1:
            for j, (ax, scatter, best_scatter) in enumerate(
                zip(param_axs, param_scatters, best_param_scatters)
            ):
                params_j = f_params[:, j] * factors
                scatter.set_offsets(np.c_[params_j, dists])
                best_scatter.set_offsets(np.c_[best_params[j], best_dist])
                ax.set_ylim(0.0, np.nanmax(dists[np.isfinite(dists)]) * 1.1)

        dh.update(fig)

    def process_energy(i):
        nonlocal f_params, sf_energies, fpts, allows
        param = f_params[i]
        a_min = max(EJb[0] / param[0], ECb[0] / param[1], ELb[0] / param[2])
        a_max = min(EJb[1] / param[0], ECb[1] / param[1], ELb[1] / param[2])
        if a_min > a_max:
            return i, np.inf, 1.0

        Bs, Cs = energy2linearform(sf_energies[i], allows)
        return i, *candidate_breakpoint_search(fpts, Bs, Cs, a_min, a_max)

    idx_bar = trange(f_params.shape[0], desc="Searching...")
    try:
        with AsyncFunc(update_plot) as async_plot:
            for i, dist, factor in Parallel(
                return_as="generator_unordered", n_jobs=n_jobs, require="sharedmem"
            )(delayed(process_energy)(i) for i in idx_bar):
                results[i] = dist, factor

                if not np.isnan(dist) and dist < best_dist:
                    # Update best result
                    best_idx = i
                    best_factor = factor
                    best_params = f_params[i] * factor
                    best_dist = dist

                # Update plot
                async_plot(i)
            else:
                idx_bar.set_description_str("Done! ")
        update_plot(best_idx)

    except KeyboardInterrupt:
        pass
    finally:
        idx_bar.close()
        plt.close(fig)  # Move plt.close(fig) inside finally block

    plt.ion()

    return best_params, fig


def fit_spectrum(flxs, fpts, init_params, allows, param_b, maxfun=1000):
    import scqubits as scq
    from scipy.optimize import minimize

    scq.settings.PROGRESSBAR_DISABLED, old = True, scq.settings.PROGRESSBAR_DISABLED

    evals_count = 0
    for lvl in allows.values():
        if not isinstance(lvl, list) or len(lvl) == 0:
            continue
        evals_count = max(evals_count, *[max(lv) for lv in lvl])
    evals_count += 1

    fluxonium = scq.Fluxonium(
        *init_params, flux=0.0, truncated_dim=evals_count, cutoff=40
    )

    pbar = tqdm(
        desc=f"({init_params[0]:.2f}, {init_params[1]:.2f}, {init_params[2]:.2f})",
        total=maxfun,
    )

    def callback(intermediate_result):
        pbar.update(1)
        if isinstance(intermediate_result, np.ndarray):
            # old version
            cur_params = intermediate_result
        else:
            cur_params = intermediate_result.x
        pbar.set_description(
            f"({cur_params[0]:.4f}, {cur_params[1]:.4f}, {cur_params[2]:.4f})"
        )

    def params2energy(flxs, params):
        nonlocal fluxonium, evals_count

        fluxonium.EJ = params[0]
        fluxonium.EC = params[1]
        fluxonium.EL = params[2]
        return fluxonium.get_spectrum_vs_paramvals(
            "flux", flxs, evals_count=evals_count, get_eigenstates=True
        ).energy_table

    # Find unique values in flxs and map original indices to unique indices
    uni_flxs, uni_idxs = np.unique(flxs, return_inverse=True)

    def loss_func(param):
        nonlocal fluxonium, flxs, uni_flxs, uni_idxs, allows, fpts

        energies = params2energy(uni_flxs, param)
        Bs, Cs = energy2linearform(energies, allows)
        fs = np.abs(Bs + Cs)[uni_idxs, :]
        return np.sum(np.sqrt(np.min(np.abs(fpts[:, None] - fs), axis=1)))

    res = minimize(
        loss_func,
        init_params,
        bounds=param_b,
        method="L-BFGS-B",
        options={"maxfun": maxfun},
        callback=callback,
    )

    pbar.close()

    scq.settings.PROGRESSBAR_DISABLED = old

    if isinstance(res, np.ndarray):  # old version
        best_params = res
    else:
        best_params = res.x

    return best_params
