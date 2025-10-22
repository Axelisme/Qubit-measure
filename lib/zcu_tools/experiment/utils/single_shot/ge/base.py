from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# from zcu_tools.utils.fitting import fit_dual_gauss, gauss_func
from zcu_tools.utils.fitting.singleshot import (
    calc_population_pdf,
    fit_singleshot,
    gauss_func,
)


def rotate(
    I_orig: np.ndarray, Q_orig: np.ndarray, theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    I_new = I_orig * np.cos(theta) - Q_orig * np.sin(theta)
    Q_new = I_orig * np.sin(theta) + Q_orig * np.cos(theta)
    return I_new, Q_new


def scatter_ge_plot(
    ax: plt.Axes,
    Ige: Tuple[np.ndarray, np.ndarray],
    Qge: Tuple[np.ndarray, np.ndarray],
    title: Optional[str] = None,
) -> None:
    Ig, Ie = Ige
    Qg, Qe = Qge
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)

    rand_gen = np.random.default_rng(42)
    downsample_idx = rand_gen.choice(
        np.arange(len(Ig)), size=min(1000, len(Ig)), replace=False
    )
    Ig, Qg = Ig[downsample_idx], Qg[downsample_idx]
    Ie, Qe = Ie[downsample_idx], Qe[downsample_idx]

    ax.scatter(Ig, Qg, label="g", color="b", marker=".", edgecolor="None", alpha=0.2)
    ax.scatter(Ie, Qe, label="e", color="r", marker=".", edgecolor="None", alpha=0.2)
    plt_params = dict(color="k", linestyle=":", marker="o", markersize=5)
    ax.plot(xg, yg, markerfacecolor="b", **plt_params)
    ax.plot(xe, ye, markerfacecolor="r", **plt_params)
    ax.set_xlabel("I [ADC levels]")
    ax.set_ylabel("Q [ADC levels]")
    ax.legend(loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.axis("equal")


def hist(
    Ig: np.ndarray,
    Ie: np.ndarray,
    numbins: int = 200,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    I_tot = np.concatenate((Ie, Ig))
    xlims = [np.min(I_tot), np.max(I_tot)]
    bins = np.linspace(xlims[0], xlims[1], numbins)
    ng, *_ = np.histogram(Ig, bins=bins, range=xlims)
    ne, *_ = np.histogram(Ie, bins=bins, range=xlims)
    ng = ng / np.sum(ng)
    ne = ne / np.sum(ne)
    if ax is not None:
        plt_params = dict(bins=bins, range=xlims, alpha=0.5)
        ax.hist(bins[:-1], color="b", weights=ng, label="g", **plt_params)
        ax.hist(bins[:-1], color="r", weights=ne, label="e", **plt_params)
        ax.set_ylabel("Counts", fontsize=14)
        ax.set_xlabel("I [ADC levels]", fontsize=14)
        ax.legend(loc="upper right")
        if title is not None:
            ax.set_title(title, fontsize=14)
    return ng, ne, bins


def fidelity_func(tp: float, tn: float, fp: float, fn: float) -> float:
    # this method calculates fidelity as (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N
    # return (tp + fn) / (tp + tn + fp + fn)
    if (tp + fn) > (tn + fp):
        return (tp + fn) / (tp + tn + fp + fn)
    else:
        return (tn + fp) / (tp + tn + fp + fn)


def calculate_fidelity(
    ng: np.ndarray, ne: np.ndarray, bins: np.ndarray
) -> Tuple[float, float]:
    cum_ng, cum_ne = np.cumsum(ng), np.cumsum(ne)
    contrast = np.abs(2 * (cum_ng - cum_ne) / (ng.sum() + ne.sum()))
    tind = contrast.argmax()
    # fid = 0.5 * (1 - ng[tind:].sum() / ng.sum() + 1 - ne[:tind].sum() / ne.sum())
    tp, fp = ng[tind:].sum(), ne[tind:].sum()
    tn, fn = ng[:tind].sum(), ne[:tind].sum()
    fid = fidelity_func(tp, tn, fp, fn)
    return fid, bins[tind]


def fitting_ge_and_plot(
    signals: np.ndarray,
    classify_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict],
    length_ratio: float,
    numbins: int = 100,
    logscale: bool = False,
) -> Tuple[float, float, float, np.ndarray]:
    Ig, Ie = signals.real
    Qg, Qe = signals.imag

    # Calculate the angle of rotation
    out_dict = classify_func(Ig, Qg, Ie, Qe)
    theta = out_dict["theta"]
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2

    # Rotate the IQ data
    Ig, Qg = rotate(Ig, Qg, theta)
    Ie, Qe = rotate(Ie, Qe, theta)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f"Rotation angle: {180 * theta / np.pi:.3f} deg")

    scatter_ge_plot(axs[0, 0], (Ig, Ie), (Qg, Qe), "Rotated")

    g_pdfs, e_pdfs, bins = hist(Ig, Ie, numbins, axs[1, 0])

    xs = bins[:-1]
    axs[0, 1].hist(xs, bins=bins, weights=g_pdfs, color="b", alpha=0.5)
    axs[1, 1].hist(xs, bins=bins, weights=e_pdfs, color="r", alpha=0.5)

    g_params, _ = fit_singleshot(xs, g_pdfs, e_pdfs, length_ratio)
    sg, se, s, p0, p_avg = g_params
    fit_g_pdfs = calc_population_pdf(xs, sg, se, s, p0, p_avg, length_ratio)
    fit_e_pdfs = calc_population_pdf(xs, sg, se, s, 1.0 - p0, p_avg, length_ratio)

    n_gg = 1.0 - p0
    n_ge = p0
    n_ee = 1.0 - p0
    n_eg = p0

    gg_fit = n_gg * gauss_func(xs, sg, s)
    ge_fit = n_ge * gauss_func(xs, se, s)
    eg_fit = n_eg * gauss_func(xs, sg, s)
    ee_fit = n_ee * gauss_func(xs, se, s)

    axs[0, 1].plot(xs, fit_g_pdfs, "k-", label="total")
    axs[0, 1].plot(xs, gg_fit, "b-", label="g")
    axs[0, 1].plot(xs, ge_fit, "b--", label="e")
    axs[0, 1].set_title(f"{n_gg:.1%} / {n_ge:.1%}", fontsize=14)
    axs[1, 1].plot(xs, fit_e_pdfs, "k-", label="total")
    axs[1, 1].plot(xs, eg_fit, "r-", label="g")
    axs[1, 1].plot(xs, ee_fit, "r--", label="e")
    axs[1, 1].set_title(f"{n_eg:.1%} / {n_ee:.1%}", fontsize=14)

    axs[1, 0].plot(xs, fit_g_pdfs, "b-", label="g")
    axs[1, 0].plot(xs, fit_e_pdfs, "r-", label="e")

    fid, threshold = calculate_fidelity(g_pdfs, e_pdfs, bins)

    title = "${F}_{ge}$"
    axs[1, 0].set_title(f"Histogram ({title}: {fid:.3%})", fontsize=14)

    for ax in axs.flat:
        ax.axvline(threshold, color="0.2", linestyle="--")

    if logscale:
        axs[0, 1].set_yscale("log")
        axs[1, 0].set_yscale("log")
        axs[1, 1].set_yscale("log")

    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    plt.tight_layout()
    plt.show()

    return (
        fid,
        threshold,
        theta * 180 / np.pi,
        np.array(
            [
                [n_gg, n_ge],
                [n_eg, n_ee],
            ]
        ),
        g_params,
    )
