from typing import Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.special import erfc

# from zcu_tools.utils.fitting import fit_dual_gauss, gauss_func
from zcu_tools.utils.fitting.singleshot import (
    calc_population_pdf,
    fit_singleshot,
    fit_singleshot_p0,
    gauss_func,
)


def rotate(
    I_orig: np.ndarray, Q_orig: np.ndarray, theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    I_new = I_orig * np.cos(theta) - Q_orig * np.sin(theta)
    Q_new = I_orig * np.sin(theta) + Q_orig * np.cos(theta)
    return I_new, Q_new


def scatter_ge_plot(
    ax: Axes,
    Ige: Tuple[np.ndarray, np.ndarray],
    Qge: Tuple[np.ndarray, np.ndarray],
    title: Optional[str] = None,
    max_points: int = 10000,
) -> None:
    Ig, Ie = Ige
    Qg, Qe = Qge

    g_center = np.median(Ig)
    e_center = np.median(Ie)

    rand_gen = np.random.default_rng(42)
    downsample_idx = rand_gen.choice(
        len(Ig), size=min(max_points, len(Ig)), replace=False
    )
    Ig, Qg = np.take(Ig, downsample_idx), np.take(Qg, downsample_idx)
    Ie, Qe = np.take(Ie, downsample_idx), np.take(Qe, downsample_idx)

    # sort Ig, Qg by Ig from high to low
    sort_g_idx = np.argsort(Ig)
    sort_e_idx = np.argsort(Ie)
    if g_center < e_center:
        sort_g_idx = sort_g_idx[::-1]
    else:
        sort_e_idx = sort_e_idx[::-1]
    Ig, Qg = Ig.take(sort_g_idx), Qg.take(sort_g_idx)
    Ie, Qe = Ie.take(sort_e_idx), Qe.take(sort_e_idx)

    # Mix the data to avoid one color covering the other
    I_all = np.concatenate((Ig, Ie))
    Q_all = np.concatenate((Qg, Qe))
    c_all = np.array(["b"] * len(Ig) + ["r"] * len(Ie))

    p = np.arange(len(I_all)).reshape(2, -1).T.flatten()
    ax.scatter(
        I_all.take(p),
        Q_all.take(p),
        c=c_all.take(p),
        marker=".",
        edgecolor="None",
        alpha=0.25,
    )
    # Plot empty data for legend
    ax.scatter([], [], label="g", color="b")
    ax.scatter([], [], label="e", color="r")
    ax.set_xlabel("I [ADC levels]")
    ax.set_ylabel("Q [ADC levels]")
    ax.legend(loc="upper right")
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.axis("equal")


def hist(
    Ig: np.ndarray,
    Ie: np.ndarray,
    numbins: Union[int, str] = "auto",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    I_tot = np.concatenate((Ie, Ig))
    xlims = (np.min(I_tot), np.max(I_tot))
    bins = np.histogram_bin_edges(I_tot, bins=numbins)
    ng, *_ = np.histogram(Ig, bins=bins, range=xlims)
    ne, *_ = np.histogram(Ie, bins=bins, range=xlims)
    ng = ng / np.sum(ng)
    ne = ne / np.sum(ne)
    if ax is not None:
        plt_params = dict(
            x=0.5 * (bins[1:] + bins[:-1]), bins=bins, range=xlims, alpha=0.5
        )
        ax.hist(color="b", weights=ng, label="g", **plt_params)  # type: ignore
        ax.hist(color="r", weights=ne, label="e", **plt_params)  # type: ignore
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


def calc_fidelity(
    ng: np.ndarray, ne: np.ndarray, bins: np.ndarray
) -> Tuple[float, float]:
    cum_ng, cum_ne = np.cumsum(ng), np.cumsum(ne)
    contrast = np.abs(2 * (cum_ng - cum_ne) / (ng.sum() + ne.sum()))
    tind = contrast.argmax()

    tp, fp = ng[tind:].sum(), ne[tind:].sum()
    tn, fn = ng[:tind].sum(), ne[:tind].sum()
    fid = fidelity_func(tp, tn, fp, fn)

    return fid, 0.5 * (bins[tind] + bins[tind + 1])


def calc_ideal_fidelity(sg: float, se: float, s: float) -> float:
    # The distance between the centers of the two Gaussians
    distance = np.abs(sg - se)

    if np.isclose(s, 0):
        return 1.0 if not np.isclose(distance, 0) else 0.5

    p_error = 0.5 * erfc(distance / (2 * np.sqrt(2) * s))

    tp = p_error
    fn = p_error
    tn = 1 - p_error
    fp = 1 - p_error

    return fidelity_func(tp=tp, tn=tn, fp=fp, fn=fn)


def fitting_ge_and_plot(
    signals: np.ndarray,
    classify_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict],
    numbins: Union[int, str] = "auto",
    logscale: bool = False,
    length_ratio: Optional[float] = None,
    init_p0: Optional[float] = None,
    avg_p: Optional[float] = None,
    align_t1: bool = True,
) -> Tuple[float, NDArray, Dict, Figure]:
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

    scatter_ge_plot(axs[0, 0], (Ig, Ie), (Qg, Qe), "Rotated")

    g_pdfs, e_pdfs, bins = hist(Ig, Ie, numbins, axs[1, 0])

    xs = 0.5 * (bins[:-1] + bins[1:])
    axs[0, 1].hist(xs, bins=bins, weights=g_pdfs, color="b", alpha=0.5)
    axs[1, 1].hist(xs, bins=bins, weights=e_pdfs, color="r", alpha=0.5)

    fixedparams = [None, None, None, init_p0, avg_p, length_ratio]
    ge_params, _ = fit_singleshot(xs, g_pdfs, e_pdfs, fixedparams=fixedparams)
    sg, se, s, p0, p_avg, length_ratio = ge_params
    (p0_g, l_ratio_g), _ = fit_singleshot_p0(
        xs, g_pdfs, init_p0=p0, ge_params=ge_params, fit_length_ratio=not align_t1
    )
    (p0_e, l_ratio_e), _ = fit_singleshot_p0(
        xs, e_pdfs, init_p0=1.0 - p0, ge_params=ge_params, fit_length_ratio=not align_t1
    )
    fit_g_pdfs = calc_population_pdf(xs, sg, se, s, p0_g, p_avg, l_ratio_g)
    fit_e_pdfs = calc_population_pdf(xs, sg, se, s, p0_e, p_avg, l_ratio_e)

    n_gg = 1.0 - p0_g
    n_ge = p0_g
    n_ee = p0_e
    n_eg = 1.0 - p0_e

    gg_fit = n_gg * gauss_func(xs, sg, s)
    ge_fit = n_ge * gauss_func(xs, se, s)
    eg_fit = n_eg * gauss_func(xs, sg, s)
    ee_fit = n_ee * gauss_func(xs, se, s)

    rotated_g_center = sg + 1j * np.median(Qg)
    rotated_e_center = se + 1j * np.median(Qe)

    plt_params = dict(linestyle=":", marker="o", markersize=5)
    axs[0, 0].plot(
        rotated_g_center.real,
        rotated_g_center.imag,
        markerfacecolor="b",
        color="r",
        **plt_params,
    )
    axs[0, 0].plot(
        rotated_e_center.real,
        rotated_e_center.imag,
        markerfacecolor="r",
        color="b",
        **plt_params,
    )
    axs[0, 0].set_xlim(np.min(bins), np.max(bins))

    axs[0, 1].plot(xs, fit_g_pdfs, "k-", label="total")
    if length_ratio != 0.0:
        axs[0, 1].plot(xs, gg_fit + ge_fit, "k--", alpha=0.3, label="ideal total")
    axs[0, 1].plot(xs, gg_fit, "b-", alpha=0.3)
    axs[0, 1].plot(xs, ge_fit, "r--", alpha=0.3)
    axs[0, 1].set_title(f"{n_gg:.1%} / {n_ge:.1%}", fontsize=14)
    axs[0, 1].legend()
    axs[1, 1].plot(xs, fit_e_pdfs, "k-", label="total")
    if length_ratio != 0.0:
        axs[1, 1].plot(xs, eg_fit + ee_fit, "k--", alpha=0.3, label="ideal total")
    axs[1, 1].plot(xs, ee_fit, "r-", alpha=0.3)
    axs[1, 1].plot(xs, eg_fit, "b--", alpha=0.3)
    axs[1, 1].set_title(f"{n_eg:.1%} / {n_ee:.1%}", fontsize=14)
    axs[1, 1].legend()

    axs[1, 0].plot(xs, fit_g_pdfs, "b-", label="g")
    axs[1, 0].plot(xs, fit_e_pdfs, "r-", label="e")

    fid, threshold = calc_fidelity(g_pdfs, e_pdfs, bins)
    ideal_fid = calc_ideal_fidelity(sg, se, s)

    axs[1, 0].set_title(
        r"${F}_{ge}: $" + f"{fid:.1%} / {1e2 * ideal_fid:.3g}%", fontsize=14
    )

    for ax in axs.flat:
        ax.axvline(threshold, color="0.2", linestyle="--")

    if logscale:
        axs[0, 1].set_yscale("log")
        axs[1, 0].set_yscale("log")
        axs[1, 1].set_yscale("log")
        y_max, y_min = 1.5 * np.max([g_pdfs, e_pdfs]), np.min([g_pdfs, e_pdfs]) + 1e-4
        axs[0, 1].set_ylim(y_min, y_max)
        axs[1, 0].set_ylim(y_min, y_max)
        axs[1, 1].set_ylim(y_min, y_max)

    if align_t1:
        fig.suptitle(f"Readout length = {length_ratio:.1f} " + r"$T_1$")
    else:
        fig.suptitle(f"Readout length = {l_ratio_g:.1f} / {l_ratio_e:.1f} " + r"$T_1$")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    return (
        fid,
        np.array(
            [
                [n_gg, n_ge],
                [n_eg, n_ee],
            ]
        ),
        {
            "ge_params": ge_params,
            "p0_g": p0_g,
            "p0_e": p0_e,
            "length_ratio_g": l_ratio_g,
            "length_ratio_e": l_ratio_e,
            "theta": theta,
            "threshold": threshold,
            "g_center": rotated_g_center * np.exp(-1j * theta),
            "e_center": rotated_e_center * np.exp(-1j * theta),
        },
        fig,
    )
