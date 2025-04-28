from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ...fitting import batch_fit_dual_gauss, gauss_func


def rotate(
    I_orig: np.ndarray, Q_orig: np.ndarray, theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate I and Q data points by a specified angle in the IQ plane.

    Parameters
    ----------
    I_orig : np.ndarray
        Original I (in-phase) data points.
    Q_orig : np.ndarray
        Original Q (quadrature) data points.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Rotated I and Q data (I_new, Q_new).
    """
    I_new = I_orig * np.cos(theta) - Q_orig * np.sin(theta)
    Q_new = I_orig * np.sin(theta) + Q_orig * np.cos(theta)
    return I_new, Q_new


def scatter_ge_plot(
    Is: Tuple[np.ndarray, np.ndarray],
    Qs: Tuple[np.ndarray, np.ndarray],
    ax: plt.Axes,
    title: Optional[str] = None,
) -> None:
    """
    Create a scatter plot of ground and excited state data in the IQ plane.

    Parameters
    ----------
    Is : Tuple[np.ndarray, np.ndarray]
        Tuple containing I (in-phase) data for ground and excited states (Ig, Ie).
    Qs : Tuple[np.ndarray, np.ndarray]
        Tuple containing Q (quadrature) data for ground and excited states (Qg, Qe).
    ax : plt.Axes
        Matplotlib axes object to plot on.
    title : Optional[str], default=None
        Title for the plot. If None, no title is set.

    Returns
    -------
    None
    """
    Ig, Ie = Is
    Qg, Qe = Qs
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
    """
    Create histograms for ground and excited state I data.

    Parameters
    ----------
    Ig : np.ndarray
        I (in-phase) data for ground state.
    Ie : np.ndarray
        I (in-phase) data for excited state.
    numbins : int, default=200
        Number of bins for the histogram.
    ax : Optional[plt.Axes], default=None
        Matplotlib axes object to plot on. If None, no plot is created.
    title : Optional[str], default=None
        Title for the plot. If None, no title is set.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Histogram data: (ng, ne, bins), where ng and ne are the histogram counts
        for ground and excited states, and bins are the bin edges.
    """
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
    """
    Calculate classification fidelity from confusion matrix values.

    This method calculates fidelity as the ratio of correctly classified samples
    to the total number of samples, choosing the higher accuracy class if imbalanced.

    Parameters
    ----------
    tp : float
        True positives (correctly classified as positive).
    tn : float
        True negatives (correctly classified as negative).
    fp : float
        False positives (incorrectly classified as positive).
    fn : float
        False negatives (incorrectly classified as negative).

    Returns
    -------
    float
        Classification fidelity (0.5-1.0).
    """
    # this method calculates fidelity as (Ngg+Nee)/N = Ngg/N + Nee/N=(0.5N-Nge)/N + (0.5N-Neg)/N = 1-(Nge+Neg)/N
    # return (tp + fn) / (tp + tn + fp + fn)
    if (tp + fn) > (tn + fp):
        return (tp + fn) / (tp + tn + fp + fn)
    else:
        return (tn + fp) / (tp + tn + fp + fn)


def calculate_fidelity(
    ng: np.ndarray, ne: np.ndarray, bins: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate optimal threshold and corresponding fidelity for state discrimination.

    Parameters
    ----------
    ng : np.ndarray
        Histogram counts for ground state.
    ne : np.ndarray
        Histogram counts for excited state.
    bins : np.ndarray
        Bin edges for the histogram.

    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - fid: Maximum achievable fidelity (0.5-1.0)
        - threshold: Optimal threshold value for state discrimination
    """
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
    plot: bool = True,
    numbins: int = 200,
) -> Tuple[float, float, float]:
    """
    Perform complete analysis of ground and excited state signals for qubit readout.

    This function performs the following steps:
    1. Extract I and Q components from ground and excited state signals
    2. Calculate optimal rotation angle using the provided classify_func
    3. Rotate the IQ data to align with the optimal measurement axis
    4. Calculate the optimal threshold and corresponding fidelity
    5. Optionally generate visualization plots

    Parameters
    ----------
    signals : np.ndarray
        Complex array of shape (2, N) containing measurement signals.
        First row should contain ground state signals, second row excited state signals.
    classify_func : Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict]
        Function that determines the optimal rotation angle from I/Q data.
        Should accept (Ig, Qg, Ie, Qe) as arguments and return a dictionary with at least a 'theta' key.
    plot : bool, default=True
        If True, generate visualization plots of the analysis results.
    numbins : int, default=200
        Number of bins for histograms.

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing:
        - fid: Maximum achievable fidelity (0.5-1.0)
        - threshold: Optimal threshold value for state discrimination
        - theta_deg: Optimal rotation angle in degrees
    """
    Ig, Ie = signals.real
    Qg, Qe = signals.imag

    # Calculate the angle of rotation
    out_dict = classify_func(Ig, Qg, Ie, Qe)
    theta = out_dict["theta"]
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2

    # Rotate the IQ data
    Ig, Qg = rotate(Ig, Qg, theta)
    Ie, Qe = rotate(Ie, Qe, theta)

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(f"Rotation angle: {180 * theta / np.pi:.3f} deg")

        scatter_ge_plot((Ig, Ie), (Qg, Qe), axs[0, 0], "Rotated")

        ng, ne, bins = hist(Ig, Ie, numbins, axs[1, 0])

        xs = bins[:-1]
        axs[0, 1].hist(xs, bins=bins, weights=ng, color="b", alpha=0.5)
        axs[1, 1].hist(xs, bins=bins, weights=ne, color="r", alpha=0.5)

        ge_params, _ = batch_fit_dual_gauss([xs, xs], [ng, ne])
        g_params, e_params = ge_params

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
    else:
        ng, ne, bins = hist(Ig, Ie, numbins)

    fid, threshold = calculate_fidelity(ng, ne, bins)

    if plot:
        title = "${F}_{ge}$"
        axs[1, 0].set_title(f"Histogram ({title}: {fid:.3%})", fontsize=14)

        for ax in axs.flat:
            ax.axvline(threshold, color="0.2", linestyle="--")

        plt.subplots_adjust(hspace=0.25, wspace=0.15)
        plt.tight_layout()
        plt.show()

    return fid, threshold, theta * 180 / np.pi  # fids: ge, gf, ef
