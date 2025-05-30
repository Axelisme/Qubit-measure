from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


from zcu_tools.notebook.util.fitting import gauss_func, fit_dual_gauss


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
    if ax is not None:
        plt_params = dict(bins=bins, range=xlims, alpha=0.5)
        ng, *_ = ax.hist(Ig, color="b", label="g", **plt_params)
        ne, *_ = ax.hist(Ie, color="r", label="e", **plt_params)
        ax.set_ylabel("Counts", fontsize=14)
        ax.set_xlabel("I [ADC levels]", fontsize=14)
        ax.legend(loc="upper right")
        if title is not None:
            ax.set_title(title, fontsize=14)
    else:
        ng, *_ = np.histogram(Ig, bins=bins, range=xlims)
        ne, *_ = np.histogram(Ie, bins=bins, range=xlims)
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
    numbins: int = 200,
) -> Tuple[float, float, float]:
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

    ng, ne, bins = hist(Ig, Ie, numbins, axs[1, 0])

    xs = bins[:-1]
    axs[0, 1].hist(xs, bins=bins, weights=ng, color="b", alpha=0.5)
    axs[1, 1].hist(xs, bins=bins, weights=ne, color="r", alpha=0.5)

    ge_params, _ = fit_dual_gauss(xs, ng + ne)
    g_params, _ = fit_dual_gauss(
        xs, ng, fixedparams=[None, *ge_params[1:3], None, *ge_params[4:6]]
    )
    e_params, _ = fit_dual_gauss(
        xs, ne, fixedparams=[None, *ge_params[1:3], None, *ge_params[4:6]]
    )

    n_gg, n_ge = g_params[0], g_params[3]
    n_eg, n_ee = e_params[0], e_params[3]
    n_gg, n_ge = n_gg / (n_gg + n_ge), n_ge / (n_gg + n_ge)
    n_eg, n_ee = n_eg / (n_eg + n_ee), n_ee / (n_eg + n_ee)

    gg_fit = gauss_func(xs, *g_params[:3])
    ge_fit = gauss_func(xs, *g_params[3:])
    eg_fit = gauss_func(xs, *e_params[:3])
    ee_fit = gauss_func(xs, *e_params[3:])
    axs[0, 1].plot(xs, gg_fit + ge_fit, "k-", label="total")
    axs[0, 1].plot(xs, gg_fit, "b-", label="g")
    axs[0, 1].plot(xs, ge_fit, "b--", label="e")
    axs[0, 1].set_title(f"{n_gg:.1%} / {n_ge:.1%}", fontsize=14)
    axs[1, 1].plot(xs, eg_fit + ee_fit, "k-", label="total")
    axs[1, 1].plot(xs, eg_fit, "r-", label="g")
    axs[1, 1].plot(xs, ee_fit, "r--", label="e")
    axs[1, 1].set_title(f"{n_eg:.1%} / {n_ee:.1%}", fontsize=14)

    axs[1, 0].plot(xs, gg_fit + ge_fit, "b-", label="g")
    axs[1, 0].plot(xs, eg_fit + ee_fit, "r-", label="e")

    fid, threshold = calculate_fidelity(ng, ne, bins)

    title = "${F}_{ge}$"
    axs[1, 0].set_title(f"Histogram ({title}: {fid:.3%})", fontsize=14)

    for ax in axs.flat:
        ax.axvline(threshold, color="0.2", linestyle="--")

    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    plt.tight_layout()
    plt.show()

    return (
        fid,
        threshold,
        theta * 180 / np.pi,
        (n_gg, n_ge, n_eg, n_ee),
    )  # fids: ge, gf, ef
