import matplotlib.pyplot as plt
import numpy as np

from . import fitting as ft
from .tools import NormalizeData, convert2max_contrast

figsize = (8, 6)


def lookback_analyze(Ts, Is, Qs, plot_fit=True, ratio: float = 0.3):
    """
    Ts: 1D array, time points
    Is: 1D array, I values
    Qs: 1D array, Q values
    """
    y = np.abs(Is + 1j * Qs)

    # find first idx where y is larger than 0.05 * max_y
    offset = Ts[np.argmax(y > ratio * np.max(y))]

    plt.figure(figsize=figsize)
    plt.plot(Ts, Is, label="I value")
    plt.plot(Ts, Qs, label="Q value")
    plt.plot(Ts, y, label="mag")
    if plot_fit:
        plt.axvline(offset, color="r", linestyle="--", label="predict_offset")
    plt.ylabel("a.u.")
    plt.xlabel("us")
    plt.legend()
    plt.show()

    return offset


def phase_analyze(fpts, signals, plot=True, plot_fit=True):
    """
    fpts: 1D array, frequency points
    signals: 1D array, signal points
    """
    phase = np.angle(signals)
    phase = np.unwrap(phase) * 180 / np.pi

    pOpt, err = ft.fit_line(fpts, phase)
    slope, offset = pOpt

    if plot:
        plt.figure(figsize=figsize)
        plt.plot(fpts, phase, label="phase")
        if plot_fit:
            plt.plot(fpts, slope * fpts + offset, label="fit")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return slope, offset


def freq_analyze(fpts, signals, asym=False, plot_fit=True, max_contrast=False):
    """
    fpts: 1D array, frequency points
    signals: 1D array, signal points
    """
    if max_contrast:
        y, _ = convert2max_contrast(signals.real, signals.imag)
    else:
        y = np.abs(signals)

    if asym:
        pOpt, err = ft.fit_asym_lor(fpts, y)
        curve = ft.asym_lorfunc(fpts, *pOpt)
    else:
        pOpt, err = ft.fitlor(fpts, y)
        curve = ft.lorfunc(fpts, *pOpt)
    freq, kappa = pOpt[3], 2 * pOpt[4]
    freq_err = np.sqrt(np.diag(err))[3]

    plt.figure(figsize=figsize)
    plt.plot(fpts, y, label="signal", marker="o", markersize=3)
    if plot_fit:
        plt.plot(fpts, curve, label=f"fit, $kappa$={kappa:.2f}")
        label = f"$f_res$ = {freq:.2f} +/- {freq_err:.2f}"
        plt.axvline(freq, color="r", ls="--", label=label)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return freq


def spectrum_analyze(flxs, fpts, amps, ratio):
    """
    flxs: 1D array, flux points
    fpts: 1D array, frequency points
    amps: 2D array, shape: (len(fpts), len(flxs))
    ratio: float, threshold to filter max points
    """
    from scipy.ndimage import gaussian_filter1d

    # use guassian filter to smooth the spectrum
    amps = gaussian_filter1d(amps, 1, axis=0)
    amps = amps - np.median(amps, axis=0, keepdims=True)
    amps = amps - np.median(amps, axis=1, keepdims=True)
    amps = np.abs(amps)

    norm_factor = np.std(amps, axis=0)
    threshold = 1.5 * np.mean(norm_factor)
    norm_factor = np.where(norm_factor > threshold, norm_factor, threshold)
    amps /= norm_factor  # (len(fpts), len(flxs))

    # find peaks
    max_ids = np.argmax(amps, axis=0)  # (len(flxs),)
    maxs = amps[max_ids, np.arange(amps.shape[1])]  # (len(flxs),)

    # select points with large contrast
    max_masks = maxs >= ratio  # (len(flxs),)
    s_flxs = flxs[max_masks]
    s_fpts = fpts[max_ids][max_masks]

    plt.figure(figsize=figsize)
    plt.imshow(
        amps,
        aspect="auto",
        origin="lower",
        extent=[flxs[0], flxs[-1], fpts[0], fpts[-1]],
    )
    plt.scatter(s_flxs, s_fpts, c="r", s=3)

    return s_flxs, s_fpts


def pdr_dep_analyze(fpts, pdrs, amps, contour=None):
    """
    fpts: 1D array, frequency points
    pdrs: 1D array, pdr values
    amps: 2D array, shape: (len(pdrs), len(fpts))
    """
    amps = NormalizeData(amps, 1)

    plt.figure(figsize=figsize)
    plt.pcolormesh(fpts, pdrs, amps)
    if contour is not None:
        plt.contour(fpts, pdrs, amps, levels=[contour])


def dispersive_analyze(
    fpts: np.ndarray,
    signals_g: np.ndarray,
    signals_e: np.ndarray,
    asym=False,
    plot_fit=True,
    use_fit=True,
):
    y_g = np.abs(signals_g)
    y_e = np.abs(signals_e)

    fig, ax = plt.subplots(2, 1, figsize=figsize)
    ax[0].plot(fpts, y_g, label="e", marker="o", markersize=3)
    ax[0].plot(fpts, y_e, label="g", marker="o", markersize=3)
    ax[1].plot(fpts, y_g - y_e)

    if asym:
        fit_func = ft.fit_asym_lor
        lor_func = ft.asym_lorfunc
    else:
        fit_func = ft.fitlor
        lor_func = ft.lorfunc

    pOpt1, err1 = fit_func(fpts, y_g)
    pOpt2, err2 = fit_func(fpts, y_e)
    freq1, kappa1 = pOpt1[3], 2 * pOpt1[4]
    freq2, kappa2 = pOpt2[3], 2 * pOpt2[4]
    err1 = np.sqrt(np.diag(err1))
    err2 = np.sqrt(np.diag(err2))

    curve1 = lor_func(fpts, *pOpt1)
    curve2 = lor_func(fpts, *pOpt2)

    if plot_fit:
        ax[0].plot(fpts, curve1, label=f"excited, $kappa$ = {kappa1:.2f}MHz")
        ax[0].plot(fpts, curve2, label=f"ground, $kappa$ = {kappa2:.2f}MHz")
        label1 = f"$f_res$ = {freq1:.2f} +/- {err1[3]:.2f}MHz"
        ax[0].axvline(freq1, color="r", ls="--", label=label1)
        label2 = f"$f_res$ = {freq2:.2f} +/- {err2[3]:.2f}MHz"
        ax[0].axvline(freq2, color="g", ls="--", label=label2)

    diff_curve = curve1 - curve2 if use_fit else y_g - y_e
    max_id = np.argmax(diff_curve)
    min_id = np.argmin(diff_curve)

    max_fpt = fpts[max_id]
    min_fpt = fpts[min_id]
    if plot_fit:
        ax[1].plot(fpts, curve1 - curve2)
        ax[1].axvline(max_fpt, color="r", ls="--", label=f"max SNR1 = {max_fpt:.2f}")
        ax[1].axvline(min_fpt, color="g", ls="--", label=f"max SNR2 = {min_fpt:.2f}")

    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    max_diff = np.abs(diff_curve[max_id])
    min_diff = np.abs(diff_curve[min_id])
    return max_fpt, min_fpt if max_diff >= min_diff else min_fpt, max_fpt


def dispersive_analyze2(fpts, signals_g, signals_e):
    y_g = np.abs(signals_g)
    y_e = np.abs(signals_e)
    y_d = np.abs(signals_g - signals_e)

    # plot signals
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    ax[0].plot(fpts, y_g, label="e", marker="o", markersize=3)
    ax[0].plot(fpts, y_e, label="g", marker="o", markersize=3)
    ax[0].legend()

    # plot difference and max/min points
    diff_curve = y_g - y_e
    max_fpt = fpts[np.argmax(diff_curve)]
    min_fpt = fpts[np.argmin(diff_curve)]
    abs_fpt = fpts[np.argmax(y_d)]
    ax[1].plot(fpts, diff_curve, label="abs", marker="o", markersize=3)
    ax[1].plot(fpts, y_d, label="iq", marker="o", markersize=3)
    ax[1].axvline(max_fpt, color="r", ls="--", label=f"max SNR1 = {max_fpt:.2f}")
    ax[1].axvline(min_fpt, color="g", ls="--", label=f"max SNR2 = {min_fpt:.2f}")
    ax[1].axvline(abs_fpt, color="b", ls="--", label=f"max Max IQ = {abs_fpt:.2f}")

    plt.tight_layout()
    plt.show()

    return abs_fpt, max_fpt, min_fpt


def rabi_analyze(
    x: int, signals: float, plot_fit=True, decay=False, max_contrast=False
):
    """
    x: 1D array, sweep points
    signals: 1D array, signal points
    """
    if max_contrast:
        y, _ = convert2max_contrast(signals.real, signals.imag)
    else:
        y = np.abs(signals)

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

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    if plot_fit:
        plt.plot(x, curve, label="fit")
        plt.axvline(pi_x, ls="--", c="red", label=f"pi={pi_x:.1f}")
        plt.axvline(pi2_x, ls="--", c="red", label=f"pi/2={(pi2_x):.1f}")
    plt.title("Rabi", fontsize=15)
    plt.legend(loc=4)
    plt.tight_layout()
    plt.show()

    return pi_x, pi2_x
