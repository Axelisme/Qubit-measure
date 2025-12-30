from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .base import fit_func, batch_fit_func


def model_func(t, T_ge, T_eg, T_eo, T_oe, T_go, T_og, pg0, pe0) -> NDArray[np.float64]:
    P0 = np.array([pg0, pe0, 1.0 - pg0 - pe0])
    M = np.array(
        [
            [-(T_ge + T_go), T_eg, T_og],
            [T_ge, -(T_eg + T_eo), T_oe],
            [T_go, T_eo, -(T_og + T_oe)],
        ]
    )

    # P(t) = exp(Mt) P(0) = V exp(Dt) V^-1 P(0)
    evals, evecs = np.linalg.eig(M)
    W = evecs * np.linalg.solve(evecs, P0)
    exp_terms = np.exp(np.outer(t, evals))
    P_t = exp_terms @ W.T

    return np.real(P_t)


def guess_initial_params(
    populations: NDArray[np.float64], times: NDArray[np.float64]
) -> Tuple[float, ...]:
    """
    使用積分法最小平方估計三能階系統速率常數。
    假設 populations 欄位順序為: [Ground (g), Excited (e), Other (o)]
    """
    # 1. 數值積分: \int P(dt)
    P_int = cumulative_trapezoid(populations, times, axis=0, initial=0)

    # 2. 差分: P(t) - P(0)
    P0 = populations[0]
    delta_P = populations - P0

    # 初始機率分布
    pg0 = max(0.0, min(1.0, P0[0]))
    pe0 = max(0.0, min(1.0, P0[1]))

    # Variable order in x: [T_ge, T_eg, T_eo, T_oe, T_go, T_og]
    # Index mapping:        0     1     2     3     4     5

    Ig = P_int[:, 0]
    Ie = P_int[:, 1]
    Io = P_int[:, 2]

    # Zeros array for padding
    Z = np.zeros_like(Ig)

    # Construct Matrix A for Ax = b
    # Equation for g: Pg - Pg0 = -Ig(T_ge + T_go) + Ie(T_eg) + Io(T_og)
    # x coeffs: [-Ig, Ie, 0, 0, -Ig, Io]
    A_g = np.stack([-Ig, Ie, Z, Z, -Ig, Io], axis=1)
    b_g = delta_P[:, 0]

    # Equation for e: Pe - Pe0 = Ig(T_ge) + Io(T_oe) - Ie(T_eg + T_eo)
    # x coeffs: [Ig, -Ie, -Ie, Io, 0, 0]
    A_e = np.stack([Ig, -Ie, -Ie, Io, Z, Z], axis=1)
    b_e = delta_P[:, 1]

    # Equation for o: Po - Po0 = Ig(T_go) + Ie(T_eo) - Io(T_og + T_oe)
    # x coeffs: [Z, Z, Ie, -Io, Ig, -Io]
    A_o = np.stack([Z, Z, Ie, -Io, Ig, -Io], axis=1)
    b_o = delta_P[:, 2]

    # Stack all equations
    A = np.vstack([A_g, A_e, A_o])
    b = np.concatenate([b_g, b_e, b_o])

    # Solve Ax = b subject to x >= 0
    x, _ = nnls(A, b)

    T_ge, T_eg, T_eo, T_oe, T_go, T_og = x

    return (T_ge, T_eg, T_eo, T_oe, T_go, T_og, pg0, pe0)


def fit_transition_rates(
    times: NDArray[np.float64], populations: NDArray[np.float64]
) -> Tuple[
    Tuple[float, ...],
    Tuple[float, ...],
    NDArray[np.float64],
    Tuple[List[float], NDArray[np.float64]],
]:
    """
    Returns:
        fitted_rates: (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        fit_populations: Fitted populations over time
        (pOpt, pCov)
    """
    pass_times = times - times[0]

    p0_guess = guess_initial_params(populations, pass_times)

    R_ge, R_eg, R_eo, R_oe, R_go, R_og, p0_g, p0_e = p0_guess

    max_R = 5 * float(np.max([R_ge, R_eg, R_eo, R_oe, R_go, R_og]))
    pOpt, pCov = fit_func(
        pass_times,
        populations.flatten(),
        lambda *args: model_func(*args).flatten(),
        p0_guess,
        bounds=(
            [0.0] * 6 + [max(0, p0_g - 0.01), max(0, p0_e - 0.01)],
            [max_R] * 6 + [min(1, p0_g + 0.01), min(1, p0_e + 0.01)],
        ),
    )
    # pOpt = list(p0_guess)
    # pCov = np.eye(len(pOpt)) * 0.01

    fit_populations = model_func(pass_times, *pOpt)

    R_ge, R_eg, R_eo, R_oe, R_go, R_og, *_ = pOpt

    rates = (R_ge, R_eg, R_eo, R_oe, R_go, R_og)
    rate_errs = tuple(np.sqrt(np.diag(pCov))[:6])

    return rates, rate_errs, fit_populations, (pOpt, pCov)


def fit_dual_transition_rates(
    times: NDArray[np.float64],
    populations1: NDArray[np.float64],
    populations2: NDArray[np.float64],
) -> Tuple[
    Tuple[float, ...],
    Tuple[float, ...],
    NDArray[np.float64],
    NDArray[np.float64],
    Tuple[List[float], NDArray[np.float64]],
    Tuple[List[float], NDArray[np.float64]],
]:
    """
    Returns:
        fitted_rates: (T_ge, T_eg, T_eo, T_oe, T_go, T_og)
        rate_errs: Errors of the fitted rates
        fit_populations1: Fitted populations1 over time
        fit_populations2: Fitted populations2 over time
        (pOpt_1, pCov_1)
        (pOpt_2, pCov_2)
    """

    pass_times = times - times[0]

    p0_guess1 = guess_initial_params(populations1, pass_times)
    p0_guess2 = guess_initial_params(populations2, pass_times)

    p0_g1, p0_e1 = p0_guess1[6], p0_guess1[7]
    p0_g2, p0_e2 = p0_guess2[6], p0_guess2[7]
    p0_guess_avg = 0.5 * (np.array(p0_guess1) + np.array(p0_guess2))

    p0_init1 = tuple([*p0_guess_avg[:6], p0_g1, p0_e1])
    p0_init2 = tuple([*p0_guess_avg[:6], p0_g2, p0_e2])

    max_R = 5 * float(np.max(p0_guess_avg[:6]))
    (pOpt1, pOpt2), (pCov1, pCov2) = batch_fit_func(
        [pass_times, pass_times],
        [populations1.flatten(), populations2.flatten()],
        lambda *args: model_func(*args).flatten(),
        [p0_init1, p0_init2],
        shared_idxs=[0, 1, 2, 3, 4, 5],
        list_bounds=[
            (
                [0.0] * 6 + [max(0.0, p0_g1 - 0.01), max(0.0, p0_e1 - 0.01)],
                [max_R] * 6 + [min(1.0, p0_g1 + 0.01), min(1.0, p0_e1 + 0.01)],
            ),
            (
                [0.0] * 6 + [max(0.0, p0_g2 - 0.01), max(0.0, p0_e2 - 0.01)],
                [max_R] * 6 + [min(1.0, p0_g2 + 0.01), min(1.0, p0_e2 + 0.01)],
            ),
        ],
    )
    # pOpt1 = list(p0_guess1)
    # pOpt2 = list(p0_guess2)
    # pCov1 = np.eye(len(pOpt1)) * 0.01
    # pCov2 = np.eye(len(pOpt2)) * 0.01

    fit_pops1 = model_func(pass_times, *pOpt1)
    fit_pops2 = model_func(pass_times, *pOpt2)

    rates = tuple(pOpt1[:6])
    rate_errs = tuple(np.sqrt(np.diag(pCov1))[:6])

    return rates, rate_errs, fit_pops1, fit_pops2, (pOpt1, pCov1), (pOpt2, pCov2)


def calc_lambdas(
    rate: Tuple[float, ...],
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    T_ge, T_eg, T_eo, T_oe, T_go, T_og = rate
    M = [
        [-(T_ge + T_go), T_eg, T_og],
        [T_ge, -(T_eg + T_eo), T_oe],
        [T_go, T_eo, -(T_og + T_oe)],
    ]
    lambdas, vectors = np.linalg.eig(M)

    lambdas = -np.real(lambdas)
    vectors = vectors.astype(np.complex128)

    sort_idxs = np.argsort(lambdas)
    lambdas = lambdas[sort_idxs]
    vectors = vectors[:, sort_idxs]

    return lambdas, vectors


def clac_amplitude(
    vectors: NDArray[np.complex128], p0g: float, p0e: float
) -> Tuple[float, float, float]:
    P0 = np.array([p0g, p0e, 1.0 - p0g - p0e])  # (3,)
    try:
        amplitudes = np.abs(np.linalg.solve(vectors, P0))  # (3,)
    except np.linalg.LinAlgError:
        amplitudes = np.array([0.0, 0.0, 0.0])
    return tuple(amplitudes)


def fit_with_vadality(times: NDArray[np.float64], populations: NDArray[np.float64]):
    pass_times = times - times[0]

    Rs = []
    R_errs = []
    fit_pps = []
    fit_lambdas = []
    fit_amplitudes = []
    time_idxs = list(range(len(times) // 2, len(times)))
    for i in tqdm(time_idxs):
        rate, rate_err, *_, (pOpt, _) = fit_transition_rates(times[:i], populations[:i])
        Rs.append(rate)
        R_errs.append(rate_err)
        fit_pps.append(model_func(pass_times, *pOpt))
        w, v = calc_lambdas(rate)
        amp = clac_amplitude(v, pOpt[6], pOpt[7])
        fit_lambdas.append(w)
        fit_amplitudes.append(amp)
    Rs = np.array(Rs)
    R_errs = np.array(R_errs)
    fit_pps = np.array(fit_pps)
    fit_lambdas = np.array(fit_lambdas)  # (N, 3)
    fit_amplitudes = np.array(fit_amplitudes)  # (N, 3)

    fig, (ax, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))

    plot_kwargs = dict(ls="-", marker=".", markersize=3)
    ax.plot(times, populations[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
    ax.plot(times, populations[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
    ax.plot(times, populations[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore

    r_num = Rs.shape[0]
    for i in range(r_num):
        ax.plot(times, fit_pps[i, :, 0], color="blue", alpha=i / r_num)  # type: ignore
        ax.plot(times, fit_pps[i, :, 1], color="red", alpha=i / r_num)  # type: ignore
        ax.plot(times, fit_pps[i, :, 2], color="green", alpha=i / r_num)  # type: ignore

    ax.legend()
    ax.grid(True)

    names = ["T_ge", "T_eg", "T_eo", "T_oe", "T_go", "T_og"]
    for i in range(Rs.shape[1]):
        ax2.errorbar(
            times[time_idxs],
            Rs[:, i],
            yerr=R_errs[:, i],
            label=names[i],
            fmt=".-",
            markersize=4,
            capsize=3,
        )
    ax2.set_ylim(0.0, 2 * np.max(Rs[-10:]))
    ax2.legend()
    ax2.grid(True)

    ax3.plot(times[time_idxs], fit_lambdas[:, 0], "o-", label="Lambda 0", markersize=4)
    ax3.plot(times[time_idxs], fit_lambdas[:, 1], "o-", label="Lambda 1", markersize=4)
    ax3.plot(times[time_idxs], fit_lambdas[:, 2], "o-", label="Lambda 2", markersize=4)
    ax3.legend()
    ax3.grid(True)

    ax4.plot(times[time_idxs], fit_amplitudes[:, 0], "o-", label="Amp 0", markersize=4)
    ax4.plot(times[time_idxs], fit_amplitudes[:, 1], "o-", label="Amp 1", markersize=4)
    ax4.plot(times[time_idxs], fit_amplitudes[:, 2], "o-", label="Amp 2", markersize=4)
    ax4.legend()
    ax4.grid(True)

    plt.show(fig)
    plt.close(fig)

    rate, rate_err, *_, (_, pCov) = fit_transition_rates(times, populations)
    pCov = pCov[:6, :6]
    for i, name in enumerate(names):
        print(
            f"{name}: {rate[i]:.4g} ± {rate_err[i]:.4g} 1/us (Rel. Error: {rate_err[i] / rate[i] * 100:.2f} %)"
        )

    fig, ax5 = plt.subplots(figsize=(8, 8))

    max_pcov = np.max(pCov)
    min_pcov = np.min(pCov)
    im = ax5.imshow(pCov, cmap="Blues", vmin=min_pcov, vmax=max_pcov)
    fig.colorbar(im, ax=ax5)

    for i in range(pCov.shape[0]):
        for j in range(pCov.shape[1]):
            val = pCov[i, j]
            ax5.text(
                j,
                i,
                f"{val:.1g}",
                ha="center",
                va="center",
                color="white" if val > 0.5 * (max_pcov + min_pcov) else "black",
            )

    ax5.set_xticks(list(range(pCov.shape[0])))
    ax5.set_yticks(list(range(pCov.shape[0])))
    ax5.set_xticklabels(names)
    ax5.set_yticklabels(names)

    plt.show(fig)
    plt.close(fig)


def fit_dual_with_vadality(
    times: NDArray[np.float64],
    populations1: NDArray[np.float64],
    populations2: NDArray[np.float64],
):
    pass_times = times - times[0]
    time_idxs = list(range(len(times) // 2, len(times)))

    Rs = []
    R_errs = []
    fit_lambdas = []
    fit_pps1 = []
    fit_pps2 = []
    fit_amps1 = []
    fit_amps2 = []
    for i in tqdm(time_idxs):
        rate, rate_err, *_, (pOpt1, _), (pOpt2, _) = fit_dual_transition_rates(
            times[:i], populations1[:i], populations2[:i]
        )
        Rs.append(rate)
        R_errs.append(rate_err)
        w, v = calc_lambdas(rate)
        fit_lambdas.append(w)
        fit_pps1.append(model_func(pass_times, *pOpt1))
        fit_pps2.append(model_func(pass_times, *pOpt2))
        fit_amps1.append(clac_amplitude(v, pOpt1[6], pOpt1[7]))
        fit_amps2.append(clac_amplitude(v, pOpt2[6], pOpt2[7]))
    Rs = np.array(Rs)
    R_errs = np.array(R_errs)
    fit_lambdas = np.array(fit_lambdas)  # (N, 3)
    fit_pps1 = np.array(fit_pps1)  # (N, times, 3)
    fit_pps2 = np.array(fit_pps2)  # (N, times, 3)
    fit_amps1 = np.array(fit_amps1)  # (N, 3)
    fit_amps2 = np.array(fit_amps2)  # (N, 3)

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 8))

    # ax1
    plot_kwargs = dict(ls="-", marker=".", markersize=3)
    ax1.plot(times, populations1[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
    ax1.plot(times, populations1[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
    ax1.plot(times, populations1[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore

    r_num = Rs.shape[0]
    for i in range(r_num):
        ax1.plot(times, fit_pps1[i, :, 0], color="blue", alpha=i / r_num)  # type: ignore
        ax1.plot(times, fit_pps1[i, :, 1], color="red", alpha=i / r_num)  # type: ignore
        ax1.plot(times, fit_pps1[i, :, 2], color="green", alpha=i / r_num)  # type: ignore

    ax1.legend()
    ax1.grid(True)

    # ax2
    plot_kwargs = dict(ls="-", marker=".", markersize=3)
    ax2.plot(times, populations2[:, 0], color="blue", label="Ground", **plot_kwargs)  # type: ignore
    ax2.plot(times, populations2[:, 1], color="red", label="Excited", **plot_kwargs)  # type: ignore
    ax2.plot(times, populations2[:, 2], color="green", label="Other", **plot_kwargs)  # type: ignore

    r_num = Rs.shape[0]
    for i in range(r_num):
        ax2.plot(times, fit_pps2[i, :, 0], color="blue", alpha=i / r_num)  # type: ignore
        ax2.plot(times, fit_pps2[i, :, 1], color="red", alpha=i / r_num)  # type: ignore
        ax2.plot(times, fit_pps2[i, :, 2], color="green", alpha=i / r_num)  # type: ignore

    ax2.legend()
    ax2.grid(True)

    # ax3
    names = ["T_ge", "T_eg", "T_eo", "T_oe", "T_go", "T_og"]
    for i in range(Rs.shape[1]):
        ax3.errorbar(
            times[time_idxs],
            Rs[:, i],
            yerr=R_errs[:, i],
            label=names[i],
            fmt=".-",
            markersize=4,
            capsize=3,
        )
    ax3.set_ylim(0.0, 2 * np.max(Rs[-10:]))
    ax3.legend()
    ax3.grid(True)

    # ax4
    ax4.plot(times[time_idxs], fit_lambdas[:, 0], "o-", label="Lambda 0", markersize=4)
    ax4.plot(times[time_idxs], fit_lambdas[:, 1], "o-", label="Lambda 1", markersize=4)
    ax4.plot(times[time_idxs], fit_lambdas[:, 2], "o-", label="Lambda 2", markersize=4)
    ax4.legend()
    ax4.grid(True)

    # ax5
    ax5.plot(times[time_idxs], fit_amps1[:, 0], "o-", label="Amp 0", markersize=4)
    ax5.plot(times[time_idxs], fit_amps1[:, 1], "o-", label="Amp 1", markersize=4)
    ax5.plot(times[time_idxs], fit_amps1[:, 2], "o-", label="Amp 2", markersize=4)
    ax5.legend()
    ax5.grid(True)

    # ax6
    ax6.plot(times[time_idxs], fit_amps2[:, 0], "o-", label="Amp 0", markersize=4)
    ax6.plot(times[time_idxs], fit_amps2[:, 1], "o-", label="Amp 1", markersize=4)
    ax6.plot(times[time_idxs], fit_amps2[:, 2], "o-", label="Amp 2", markersize=4)
    ax6.legend()
    ax6.grid(True)

    plt.show(fig)
    plt.close(fig)

    rate, rate_err, *_, (_, pCov2) = fit_dual_transition_rates(
        times, populations1, populations2
    )
    for i, name in enumerate(names):
        err_perc = rate_err[i] / rate[i] * 100 if rate[i] != 0 else 0.0
        print(
            f"{name}: {rate[i]:.4g} ± {rate_err[i]:.4g} us^-1 (Rel. Error: {err_perc:.2f} %)"
        )

    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    pCov = pCov2[:6, :6]
    rpCov = pCov.copy()
    for i in range(rpCov.shape[0]):
        for j in range(rpCov.shape[1]):
            rpCov[i, j] = pCov[i, j] / (np.sqrt(pCov[i, i] * pCov[j, j]) + 1e-12)

    max_val = np.max(np.abs(rpCov))
    ax1.imshow(rpCov, vmin=-max_val, vmax=max_val, cmap="viridis")
    for i in range(rpCov.shape[0]):
        for j in range(rpCov.shape[1]):
            val = rpCov[i, j]
            ax1.text(
                j,
                i,
                f"{val:.1g}",
                ha="center",
                va="center",
                color="white" if val < 0 else "black",
            )

    ax1.set_xticks(list(range(rpCov.shape[0])))
    ax1.set_yticks(list(range(rpCov.shape[0])))
    ax1.set_xticklabels(names)
    ax1.set_yticklabels(names)

    max_val = np.max(np.abs(pCov))
    ax2.imshow(pCov, vmin=-max_val, vmax=max_val, cmap="viridis")
    for i in range(pCov.shape[0]):
        for j in range(pCov.shape[1]):
            val = pCov[i, j]
            ax2.text(
                j,
                i,
                f"{val:.1g}",
                ha="center",
                va="center",
                color="white" if val < 0 else "black",
            )

    ax2.set_xticks(list(range(pCov.shape[0])))
    ax2.set_yticks(list(range(pCov.shape[0])))
    ax2.set_xticklabels(names)
    ax2.set_yticklabels(names)

    plt.show(fig)
