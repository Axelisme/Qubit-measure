import matplotlib.pyplot as plt
import numpy as np

from . import fitting as ft
from .tools import convert2max_contrast

figsize = (8, 6)


def lookback_analyze(Ts, Is, Qs, plot=True, ratio: float = 0.3):
    y = np.abs(Is + 1j * Qs)

    # find first idx where y is larger than 0.05 * max_y
    offset = Ts[np.argmax(y > ratio * np.max(y))]

    if plot:
        plt.figure(figsize=figsize)
        plt.plot(Ts, Is, label="I value")
        plt.plot(Ts, Qs, label="Q value")
        plt.plot(Ts, y, label="mag")
        plt.axvline(offset, color="r", linestyle="--", label="predict_offset")
        plt.ylabel("a.u.")
        plt.xlabel("us")
        plt.legend()

    return offset


def phase_analyze(x, y, plot=True, plot_fit=True):
    phase = np.angle(y)
    phase = np.unwrap(phase) * 180 / np.pi

    pOpt, err = ft.fit_line(x, phase)
    slope, offset = pOpt

    if plot:
        plt.figure(figsize=figsize)
        plt.plot(x, phase, label="phase")
        if plot_fit:
            plt.plot(x, slope*x+offset, label="fit")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return slope, offset


def freq_analyze(
    x, y, asym=False, plot=True, show_center=True, max_contrast=False
):
    if max_contrast:
        signal, _ = convert2max_contrast(y.real, y.imag)
    else:
        signal = np.abs(y)

    if asym:
        pOpt, err = ft.fit_asym_lor(x, signal)
        curve = ft.asym_lorfunc(x, *pOpt)
    else:
        pOpt, err = ft.fitlor(x, signal)
        curve = ft.lorfunc(x, *pOpt)
    freq, kappa = pOpt[3], 2 * pOpt[4]
    freq_err = np.sqrt(np.diag(err))[3]

    if plot:
        plt.figure(figsize=figsize)
        plt.plot(x, signal, label="signal", marker="o", markersize=3)
        plt.plot(x, curve, label=f"fit, $kappa$={kappa:.2f}")
        if show_center:
            label = f"$f_res$ = {freq:.2f} +/- {freq_err:.2f}"
            plt.axvline(freq, color="r", ls="--", label=label)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return freq


def dispersive_analyze(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    use_fit=True,
    asym=False,
    plot=True,
    max_contrast=False,
):
    if max_contrast:
        y1, _ = convert2max_contrast(y1.real, y1.imag)
        y2, _ = convert2max_contrast(y2.real, y2.imag)
    else:
        y1 = np.abs(y1)
        y2 = np.abs(y2)

    if asym:
        fit_func = ft.fit_asym_lor
        lor_func = ft.asym_lorfunc
    else:
        fit_func = ft.fitlor
        lor_func = ft.lorfunc

    pOpt1, err1 = fit_func(x, y1)
    pOpt2, err2 = fit_func(x, y2)
    freq1, kappa1 = pOpt1[3], 2 * pOpt1[4]
    freq2, kappa2 = pOpt2[3], 2 * pOpt2[4]
    err1 = np.sqrt(np.diag(err1))
    err2 = np.sqrt(np.diag(err2))

    curve1 = lor_func(x, *pOpt1)
    curve2 = lor_func(x, *pOpt2)
    if use_fit:
        diff_curve = curve1 - curve2
    else:
        diff_curve = y1 - y2
    max_id = np.argmax(diff_curve)
    min_id = np.argmin(diff_curve)

    if plot:
        plt.figure(figsize=figsize)
        plt.title(f"$chi=${(freq2-freq1):.3f}, unit = MHz", fontsize=15)
        plt.plot(x, y1, label="e", marker="o", markersize=3)
        plt.plot(x, y2, label="g", marker="o", markersize=3)
        plt.plot(x, curve1, label=f"fite, $kappa$ = {kappa1:.2f}MHz")
        plt.plot(x, curve2, label=f"fitg, $kappa$ = {kappa2:.2f}MHz")
        label1 = f"$f_res$ = {freq1:.2f} +/- {err1[3]:.2f}MHz"
        plt.axvline(freq1, color="r", ls="--", label=label1)
        label2 = f"$f_res$ = {freq2:.2f} +/- {err2[3]:.2f}MHz"
        plt.axvline(freq2, color="g", ls="--", label=label2)
        plt.legend()

        plt.figure(figsize=figsize)
        plt.plot(x, y1 - y2)
        plt.plot(x, curve1 - curve2)
        plt.axvline(x[max_id], color="r", ls="--", label=f"max SNR1 = {x[max_id]:.2f}")
        plt.axvline(x[min_id], color="g", ls="--", label=f"max SNR2 = {x[min_id]:.2f}")
        plt.legend()
        plt.show()

    if np.abs(diff_curve[max_id]) >= np.abs(diff_curve[min_id]):
        return x[max_id], x[min_id]
    else:
        return x[min_id], x[max_id]


def rabi_analyze(x: int, y: float, plot=True, decay=False, max_contrast=False):
    if max_contrast:
        y, _ = convert2max_contrast(y.real, y.imag)
    else:
        y = np.abs(y)

    if decay:
        fit_func = ft.fitdecaysin
        sin_func = ft.decaysin
    else:
        fit_func = ft.fitsin
        sin_func = ft.sinfunc

    pOpt, _ = fit_func(x, y)
    curve = sin_func(x, *pOpt)

    freq = pOpt[2]
    phase = pOpt[3] % 360 - 180
    if phase < 0:
        pi_x = (0.25 - phase / 360) / freq
        pi2_x = -phase / 360 / freq
    else:
        pi_x = (0.75 - phase / 360) / freq
        pi2_x = (0.5 - phase / 360) / freq

    if plot:
        plt.figure(figsize=figsize)
        plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
        plt.plot(x, curve, label="fit")
        plt.title("Rabi", fontsize=15)
        plt.axvline(pi_x, ls="--", c="red", label=f"$\pi$={pi_x:.1f}")
        plt.axvline(pi2_x, ls="--", c="red", label=f"$\pi/2$={(pi2_x):.1f}")
        plt.legend(loc=4)
        plt.tight_layout()
        plt.show()

    return pi_x, pi2_x
