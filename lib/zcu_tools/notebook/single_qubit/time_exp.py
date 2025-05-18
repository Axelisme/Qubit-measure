from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import zcu_tools.notebook.util.fitting as ft

from .general import figsize
from .process import rotate2real


def rabi_analyze(
    xs: np.ndarray,
    signals: np.ndarray,
    plot_fit=True,
    decay=False,
    max_contrast=False,
    xlabel=None,
) -> Tuple[float, float]:
    """
    Analyzes Rabi oscillation measurements and extracts pi and pi/2 pulse times.

    Parameters
    ----------
    xs : np.ndarray
        1D array of time (or other sweep parameter) values
    signals : np.ndarray
        1D array of complex measurement signals corresponding to xs values
    plot_fit : bool, optional
        If True, plots the fit results. Default is True.
    decay : bool, optional
        If True, uses a decaying cosine fit function. Default is False.
    max_contrast : bool, optional
        If True, rotates the signal to the real axis before analysis. Default is False.
    xlabel : str, optional
        Label for the x-axis on the plot. Default is None.

    Returns
    -------
    Tuple[float, float]
        pi_x: Duration for a pi pulse
        pi2_x: Duration for a pi/2 pulse
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

    freq: float = pOpt[2]
    phase: float = pOpt[3] % 360
    if phase > 270:
        pi_x = (1.5 - phase / 360) / freq
        pi2_x = (1.25 - phase / 360) / freq
    elif phase < 90:
        pi_x = (0.5 - phase / 360) / freq
        pi2_x = (0.25 - phase / 360) / freq
    else:
        pi_x = (1.0 - phase / 360) / freq
        pi2_x = (0.75 - phase / 360) / freq
    assert isinstance(pi_x, float) and isinstance(pi2_x, float)

    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.plot(xs, y, label="meas", ls="-", marker="o", markersize=3)
    if plot_fit:
        xs = np.linspace(pi_x - 0.5 / freq, xs[-1], 1000)
        curve = cos_func(xs, *pOpt)
        plt.plot(xs, curve, label="fit")
        plt.axvline(pi_x, ls="--", c="red", label=f"pi={pi_x:.2g}")
        plt.axvline(pi2_x, ls="--", c="red", label=f"pi/2={(pi2_x):.2g}")
    if xlabel is not None:
        plt.xlabel(xlabel)
    if max_contrast:
        plt.ylabel("Signal Real (a.u.)")
    else:
        plt.ylabel("Magnitude (a.u.)")
    plt.legend(loc=4)

    return pi_x, pi2_x


def T1_analyze(
    xs: np.ndarray,
    signals: np.ndarray,
    plot=True,
    max_contrast=False,
    dual_exp=False,
):
    """
    Analyzes T1 relaxation measurements and extracts the relaxation time.

    Parameters
    ----------
    xs : np.ndarray
        1D array of delay times in microseconds
    signals : np.ndarray
        1D array of complex measurement signals corresponding to xs values
    plot : bool, optional
        If True, plots the fit results. Default is True.
    max_contrast : bool, optional
        If True, rotates the signal to the real axis before analysis. Default is False.
    dual_exp : bool, optional
        If True, fits to a dual exponential model for two relaxation mechanisms.
        Default is False.

    Returns
    -------
    float
        T1: Relaxation time in microseconds
    float
        T1err: Error in the relaxation time
    """
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
        if pOpt[4] > pOpt[2]:
            pOpt = [pOpt[0], pOpt[3], pOpt[4], pOpt[1], pOpt[2]]
            err = [err[0], err[3], err[4], err[1], err[2]]
        t1b: float = pOpt[4]
        t1berr: float = err[4]
    t1: float = pOpt[2]
    t1err: float = err[2]

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

    return t1, t1err


def T2fringe_analyze(
    xs: np.ndarray, signals: np.ndarray, plot=True, max_contrast=False
):
    """
    Analyzes T2 Ramsey fringe measurements and extracts dephasing time and frequency detuning.

    Parameters
    ----------
    xs : np.ndarray
        1D array of delay times in microseconds
    signals : np.ndarray
        1D array of complex measurement signals corresponding to xs values
    plot : bool, optional
        If True, plots the fit results. Default is True.
    max_contrast : bool, optional
        If True, rotates the signal to the real axis before analysis. Default is False.

    Returns
    -------
    float
        T2f: Dephasing time in microseconds
    float
        detune: Frequency detuning in MHz
    float
        T2f_err: Error in the dephasing time
    float
        detune_err: Error in the detuning frequency
    """
    if max_contrast:
        signals = rotate2real(signals)
        y = signals.real
    else:
        y = np.abs(signals)

    pOpt, pCov = ft.fitdecaycos(xs, y)
    t2f: float = pOpt[4]
    detune: float = pOpt[2]
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

    return t2f, detune, err[4], err[2]


def T2decay_analyze(xs: np.ndarray, signals: np.ndarray, plot=True, max_contrast=False):
    """
    Analyzes T2 echo decay measurements and extracts the coherence time.

    Parameters
    ----------
    xs : np.ndarray
        1D array of delay times in microseconds
    signals : np.ndarray
        1D array of complex measurement signals corresponding to xs values
    plot : bool, optional
        If True, plots the fit results. Default is True.
    max_contrast : bool, optional
        If True, rotates the signal to the real axis before analysis. Default is False.

    Returns
    -------
    float
        T2e: Echo coherence time in microseconds
    float
        T2e_err: Error in the coherence time
    """
    if max_contrast:
        signals = rotate2real(signals)
        y = signals.real
    else:
        y = np.abs(signals)

    pOpt, pCov = ft.fitexp(xs, y)
    t2e: float = pOpt[2]
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

    return t2e, err[2]
