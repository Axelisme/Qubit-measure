from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from . import fitting as ft

figsize = (8, 6)


def lookback_analyze(x: np.ndarray, Is: np.ndarray, Qs: np.ndarray):
    y = np.abs(Is + 1j * Qs)

    # find first idx where y is larger than 0.05 * max_y
    idx = np.argmax(y > 0.05 * np.max(y))

    return x[idx]


def freq_analyze(x, y, asym=False, plot=True):
    mag = np.abs(y)
    pha = np.unwrap(np.angle(y))
    if asym:
        pOpt_mag, _ = ft.fit_asym_lor(x, mag)
        pOpt_pha, _ = ft.fit_asym_lor(x, pha)
        curve_mag = ft.asym_lorfunc(x, *pOpt_mag)
        curve_pha = ft.asym_lorfunc(x, *pOpt_pha)
    else:
        pOpt_mag, _ = ft.fitlor(x, mag)
        pOpt_pha, _ = ft.fitlor(x, pha)
        curve_mag = ft.lorfunc(x, *pOpt_mag)
        curve_pha = ft.lorfunc(x, *pOpt_pha)
    res_mag, kappa_mag = pOpt_mag[3], 2 * pOpt_mag[4]
    res_pha, kappa_pha = pOpt_pha[3], 2 * pOpt_pha[4]

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        axs[0].plot(x, mag, label="mag", marker="o", markersize=3)
        axs[0].plot(x, curve_mag, label=f"fit, $kappa$={kappa_mag:.2f}")
        axs[0].axvline(res_mag, color="r", ls="--", label=f"$f_res$ = {res_mag:.2f}")
        axs[0].set_title("mag.", fontsize=15)
        axs[0].legend()

        axs[1].plot(x, pha, label="pha", marker="o", markersize=3)
        axs[1].plot(x, curve_pha, label=f"fit, $kappa$={kappa_pha:.2f}")
        axs[1].axvline(res_pha, color="r", ls="--", label=f"$f_res$ = {res_pha:.2f}")
        axs[1].set_title("pha.", fontsize=15)
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    return round(res_mag, 2), round(res_pha, 2)


def NormalizeData(signals2D: np.ndarray) -> np.ndarray:
    # normalize on frequency axis
    mins = np.min(signals2D, axis=1, keepdims=True)
    maxs = np.max(signals2D, axis=1, keepdims=True)
    norm_const = np.clip(maxs - mins, 1e-10, None)
    return (signals2D - mins) / norm_const


def spectrum_analyze(
    fpts: np.ndarray,
    ypts: np.ndarray,
    signal2D: np.ndarray,
    f_axis: Literal["x-axis", "y-axis"] = "x-axis",
    plot_peak=True,
):
    signal2D = NormalizeData(np.abs(signal2D))

    freqs = np.zeros_like(ypts)
    pOpt = None
    for i in range(len(freqs)):
        pOpt, _ = ft.fitlor(fpts, signal2D[i], pOpt)
        freqs[i] = pOpt[3]
    freqs = np.array(freqs)

    plt.figure(figsize=figsize)
    if f_axis == "y-axis":
        # let frequency be y-axis
        plt.pcolormesh(ypts, fpts, signal2D.T, shading="auto")
        if plot_peak:
            plt.plot(ypts, freqs, color="r", marker="o", markersize=3)
    elif f_axis == "x-axis":
        # let frequency be x-axis
        plt.pcolormesh(fpts, ypts, signal2D, shading="auto")
        if plot_peak:
            plt.plot(freqs, ypts, color="r", marker="o", markersize=3)
    else:
        raise ValueError("f_axis must be 'x-axis' or 'y-axis'")
    plt.show()

    return freqs


def dispersive_analyze(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, asym=False):
    y1 = np.abs(y1)  # type: ignore
    y2 = np.abs(y2)  # type: ignore
    if asym:
        pOpt1, _ = ft.fit_asym_lor(x, y1)
        pOpt2, _ = ft.fit_asym_lor(x, y2)
        curve1 = ft.asym_lorfunc(x, *pOpt1)
        curve2 = ft.asym_lorfunc(x, *pOpt2)
    else:
        pOpt1, _ = ft.fitlor(x, y1)
        pOpt2, _ = ft.fitlor(x, y2)
        curve1 = ft.lorfunc(x, *pOpt1)
        curve2 = ft.lorfunc(x, *pOpt2)
    res1, kappa1 = pOpt1[3], 2 * pOpt1[4]
    res2, kappa2 = pOpt2[3], 2 * pOpt2[4]

    plt.figure(figsize=figsize)
    plt.title(f"$chi=${(res2-res1):.3f}, unit = MHz", fontsize=15)
    plt.plot(x, y1, label="e", marker="o", markersize=3)
    plt.plot(x, y2, label="g", marker="o", markersize=3)
    plt.plot(x, curve1, label=f"fite, $kappa$ = {kappa1:.2f}MHz")
    plt.plot(x, curve2, label=f"fitg, $kappa$ = {kappa2:.2f}MHz")
    plt.axvline(res1, color="r", ls="--", label=f"$f_res$ = {res1:.2f}")
    plt.axvline(res2, color="g", ls="--", label=f"$f_res$ = {res2:.2f}")
    plt.legend()

    plt.figure(figsize=figsize)
    plt.plot(x, y1 - y2)
    diff_curve = curve1 - curve2
    max_id = np.argmax(diff_curve)
    min_id = np.argmin(diff_curve)
    plt.plot(x, diff_curve)
    plt.axvline(
        x[np.argmax(diff_curve)],  # type: ignore
        color="r",
        ls="--",
        label=f"max SNR1 = {x[max_id]:.2f}",  # type: ignore
    )
    plt.axvline(
        x[np.argmin(curve1 - curve2)],  # type: ignore
        color="g",
        ls="--",
        label=f"max SNR2 = {x[max_id]:.2f}",
    )
    plt.legend()
    plt.show()

    if np.abs(diff_curve[max_id]) >= np.abs(diff_curve[min_id]):
        return x[max_id], x[min_id]
    else:
        return x[min_id], x[max_id]


def amprabi_analyze(x: int, y: float):
    y = np.abs(y)
    pOpt, _ = ft.fitsin(x, y)

    freq = pOpt[2]
    phase = pOpt[3] % 360 - 180
    if phase < 0:
        pi_gain = (0.25 - phase / 360) / freq
        pi2_gain = -phase / 360 / freq
    else:
        pi_gain = (0.75 - phase / 360) / freq
        pi2_gain = (0.5 - phase / 360) / freq

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    plt.plot(x, ft.sinfunc(x, *pOpt), label="fit")
    plt.title("Amplitude Rabi", fontsize=15)
    plt.xlabel("$gain$", fontsize=15)
    plt.axvline(pi_gain, ls="--", c="red", label=f"$\pi$ gain={pi_gain:.1f}")
    plt.axvline(pi2_gain, ls="--", c="red", label=f"$\pi/2$ gain={(pi2_gain):.1f}")
    plt.legend(loc=4)
    plt.tight_layout()
    plt.show()

    return pi_gain, pi2_gain, np.max(y) - np.min(y)
