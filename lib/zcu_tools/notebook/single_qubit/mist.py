from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from zcu_tools.notebook.single_qubit.process import rotate2real

from .general import figsize


def analyze_mist_pdr_dep(
    pdrs: np.ndarray, signals: np.ndarray, g0=None, e0=None, ac_coeff=None
) -> None:
    signals = rotate2real(signals)

    if g0 is None:
        g0 = signals[0]

    amp_diff = np.abs(signals - g0)

    if ac_coeff is None:
        xs = pdrs
        xlabel = "probe gain (a.u.)"
    else:
        xs = ac_coeff * pdrs**2
        xlabel = r"$\image.pngbar n$"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, amp_diff.T, marker=".")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=12)
    if e0 is not None:
        ax.set_ylim(0, 1.1 * np.abs(g0 - e0))


def analyze_abnormal_pdr_dep(
    pdrs: np.ndarray, signals: np.ndarray, g0=None, e0=None, ac_coeff=None
) -> None:
    signals = rotate2real(signals)

    signals_g = signals[0, ...]
    signals_e = signals[1, ...]

    if g0 is None:
        g0 = signals_g[0]
    if e0 is None:
        e0 = signals_e[0]
    ge_mean = (g0 + e0) / 2

    abnormal_signals = np.mean(signals, axis=0) - ge_mean

    if ac_coeff is None:
        xs = pdrs
        xlabel = "probe gain (a.u.)"
    else:
        xs = ac_coeff * pdrs**2
        xlabel = r"$\bar n$"

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(xs, np.abs(signals_g - g0), label="signal", color="blue")
    ax.plot(xs, np.abs(abnormal_signals), label="abnormal", color="green")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
    ax.set_ylim(0.0, 1.1 * np.abs(g0 - e0))
    ax.legend(fontsize="x-large")
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()
    plt.show()


def analyze_mist_pdr_overnight(
    pdrs: np.ndarray, signals: np.ndarray, pi_signal=None, ac_coeff: float = None
) -> complex:
    signals = rotate2real(signals)

    g0 = np.mean(signals[:, 0])

    abs_diff = np.abs(signals - g0)

    if ac_coeff is None:
        xs = pdrs
        xlabel = "probe gain (a.u.)"
    else:
        xs = ac_coeff * pdrs**2
        xlabel = r"$\bar n$"

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, abs_diff.T)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Signal difference (a.u.)", fontsize=14)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=12)
    if pi_signal is not None:
        ax.set_ylim(0, 1.1 * np.abs(g0 - pi_signal))

    plt.tight_layout()
    plt.show()

    return g0


def analyze_mist_flx_pdr(
    flxs: np.ndarray, pdrs: np.ndarray, signals: np.ndarray, ac_coeff: float = None
) -> Tuple[plt.Figure, plt.Axes]:
    amp_diff = np.abs(signals - signals[:, 0][:, None])
    # amp_diff = amp_diff > 0.03
    amp_diff = -np.clip(amp_diff, 0.01, 0.2)

    fig, ax = plt.subplots(figsize=figsize)

    if ac_coeff is None:
        ys = pdrs
        ylabel = "probe gain (a.u.)"
        ax.set_ylim(0.01, 1)
    else:
        ys = ac_coeff * pdrs**2
        ylabel = r"$\bar n$"
        ax.set_ylim(2, np.max(ys))

    ax.imshow(
        amp_diff.T,
        origin="lower",
        interpolation="none",
        aspect="auto",
        extent=[flxs[0], flxs[-1], ys[0], ys[-1]],
    )
    ax.set_xlabel(r"$\phi$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()

    return fig, ax
