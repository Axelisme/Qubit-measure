import matplotlib.pyplot as plt
import numpy as np

from . import fitting as ft
from .general import figsize
from .tools import rotate2real


def rabi_analyze(
    xs: np.ndarray,
    signals: np.ndarray,
    plot_fit=True,
    decay=False,
    max_contrast=False,
    xlabel=None,
):
    """
    x: 1D array, sweep points
    signals: 1D array, signal points
    """
    if max_contrast:
        signals = rotate2real(signals)
        y = signals.real
    else:
        y = np.abs(signals)

    if decay:
        fit_func = ft.fitdecaycos
        cos_func = ft.decaycos
    else:
        fit_func = ft.fitcos
        cos_func = ft.cosfunc

    pOpt, _ = fit_func(xs, y)

    freq = pOpt[2]
    phase = pOpt[3] % 360  # type: ignore
    if phase > 270:
        pi_x = (1.5 - phase / 360) / freq
        pi2_x = (1.25 - phase / 360) / freq
    elif phase < 90:
        pi_x = (0.5 - phase / 360) / freq
        pi2_x = (0.25 - phase / 360) / freq
    else:
        pi_x = (1.0 - phase / 360) / freq
        pi2_x = (0.75 - phase / 360) / freq

    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.plot(xs, y, label="meas", ls="-", marker="o", markersize=3)
    if plot_fit:
        xs = np.linspace(pi_x - 0.5 / freq, xs[-1], 1000)  # type: ignore
        curve = cos_func(xs, *pOpt)
        plt.plot(xs, curve, label="fit")
        plt.axvline(pi_x, ls="--", c="red", label=f"pi={pi_x:.1f}")
        plt.axvline(pi2_x, ls="--", c="red", label=f"pi/2={(pi2_x):.1f}")
    if xlabel is not None:
        plt.xlabel(xlabel)
    if max_contrast:
        plt.ylabel("Signal Real (a.u.)")
    else:
        plt.ylabel("Magnitude (a.u.)")
    plt.legend(loc=4)
    plt.show()

    return pi_x, pi2_x


def T1_analyze(
    xs: np.ndarray,
    signals: np.ndarray,
    return_err=False,
    plot=True,
    max_contrast=False,
    dual_exp=False,
):
    if max_contrast:
        signals = rotate2real(signals)
        y = signals.real
    else:
        y = np.abs(signals)

    if dual_exp:
        pOpt, pCov = ft.fit_dualexp(xs, y)
        sim = ft.dual_expfunc(xs, *pOpt)
    else:
        pOpt, pCov = ft.fitexp(xs, y)
        sim = ft.expfunc(xs, *pOpt)
    err = np.sqrt(np.diag(pCov))

    if dual_exp:
        # make sure t1 is the longer one
        if pOpt[4] > pOpt[2]:  # type: ignore
            pOpt = [pOpt[0], pOpt[3], pOpt[4], pOpt[1], pOpt[2]]
            err = [err[0], err[3], err[4], err[1], err[2]]
        t1b, t1berr = pOpt[4], err[4]
    t1, t1err = pOpt[2], err[2]

    if plot:
        t1_str = f"{t1:.2f}us +/- {t1err:.2f}us"
        if dual_exp:
            t1b_str = f"{t1b:.2f}us +/- {t1berr:.2f}us"

        plt.figure(figsize=figsize)
        plt.plot(xs, y, label="meas", ls="-", marker="o", markersize=3)
        plt.plot(xs, sim, label="fit")
        if dual_exp:
            plt.plot(xs, ft.expfunc(xs, *pOpt[:3]), linestyle="--", label="t1b fit")
            plt.title(f"T1 = {t1_str}, T1b = {t1b_str}", fontsize=15)
        else:
            plt.title(f"T1 = {t1_str}", fontsize=15)
        plt.xlabel("Time (us)")
        if max_contrast:
            plt.ylabel("Signal Real (a.u.)")
        else:
            plt.ylabel("Magnitude (a.u.)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if return_err:
        return t1, t1err
    return t1


def T2fringe_analyze(
    xs: np.ndarray, signals: np.ndarray, return_err=False, plot=True, max_contrast=False
):
    if max_contrast:
        signals = rotate2real(signals)
        y = signals.real
    else:
        y = np.abs(signals)

    pOpt, pCov = ft.fitdecaycos(xs, y)
    t2f, detune = pOpt[4], pOpt[2]
    sim = ft.decaycos(xs, *pOpt)
    err = np.sqrt(np.diag(pCov))

    if plot:
        t2f_str = f"{t2f:.2f}us +/- {err[4]:.2f}us"
        detune_str = f"{detune:.2f}MHz +/- {err[2] * 1e3:.2f}kHz"

        plt.figure(figsize=figsize)
        plt.plot(xs, y, label="meas", ls="-", marker="o", markersize=3)
        plt.plot(xs, sim, label="fit")
        plt.title(f"T2 fringe = {t2f_str}, detune = {detune_str}", fontsize=15)
        plt.xlabel("Time (us)")
        if max_contrast:
            plt.ylabel("Signal Real (a.u.)")
        else:
            plt.ylabel("Magnitude (a.u.)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if return_err:
        return t2f, detune, err[4], err[2]
    return t2f, detune


def T2decay_analyze(
    xs: np.ndarray, signals: np.ndarray, return_err=False, plot=True, max_contrast=False
):
    if max_contrast:
        signals = rotate2real(signals)
        y = signals.real
    else:
        y = np.abs(signals)

    pOpt, pCov = ft.fitexp(xs, y)
    t2e = pOpt[2]
    err = np.sqrt(np.diag(pCov))

    if plot:
        t2e_str = f"{t2e:.2f}us +/- {err[2]:.2f}us"

        plt.figure(figsize=figsize)
        plt.plot(xs, y, label="meas", ls="-", marker="o", markersize=3)
        plt.plot(xs, ft.expfunc(xs, *pOpt), label="fit")
        plt.title(f"T2 decay = {t2e_str}", fontsize=15)
        plt.xlabel("Time (us)")
        if max_contrast:
            plt.ylabel("Signal Real (a.u.)")
        else:
            plt.ylabel("Magnitude (a.u.)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if return_err:
        return t2e, err[2]
    return t2e
