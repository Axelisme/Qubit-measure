import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from . import fitting as ft
from .tools import NormalizeData, convert2max_contrast

figsize = (8, 6)


def lookback_analyze(
    Ts, Is, Qs, plot_fit=True, ratio: float = 0.3, pulse_cfg: dict = None
):
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
    if pulse_cfg is not None:
        trig_offset = pulse_cfg["trig_offset"]
        ro_length = pulse_cfg["ro_length"]
        plt.axvline(trig_offset, color="g", linestyle="--", label="ro start")
        plt.axvline(trig_offset + ro_length, color="g", linestyle="--", label="ro end")

    plt.xlabel("Time (us)")
    plt.ylabel("a.u.")
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
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Phase (degree)")
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
    plt.xlabel("Frequency (MHz)")
    if max_contrast:
        plt.ylabel("Signal Real (a.u.)")
    else:
        plt.ylabel("Magnitude (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return freq


def pdr_dep_analyze(fpts, pdrs, amps, contour=None):
    """
    fpts: 1D array, frequency points
    pdrs: 1D array, pdr values
    amps: 2D array, shape: (len(pdrs), len(fpts))
    """
    amps = NormalizeData(amps, 1)

    plt.figure(figsize=figsize)
    plt.imshow(
        amps,
        aspect="auto",
        origin="lower",
        extent=[fpts[0], fpts[-1], pdrs[0], pdrs[-1]],
    )
    if contour is not None:
        plt.contour(fpts, pdrs, amps, levels=[contour])


def dispersive_analyze(fpts, signals_g, signals_e):
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

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return abs_fpt, max_fpt, min_fpt


def dispersive2D_analyze(fpts, pdrs, signals2D):
    amps = np.abs(signals2D)

    amps = gaussian_filter(amps, 1)

    fpt_max_id = np.argmax(np.max(amps, axis=0))
    pdr_max_id = np.argmax(np.max(amps, axis=1))
    fpt_max = fpts[fpt_max_id]
    pdr_max = pdrs[pdr_max_id]

    plt.figure(figsize=figsize)
    plt.imshow(
        amps,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[fpts[0], fpts[-1], pdrs[0], pdrs[-1]],
    )
    plt.scatter(
        fpt_max, pdr_max, color="r", label=f"max SNR = {fpt_max:.2f}, {int(pdr_max)}"
    )
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("SNR (a.u.)")
    plt.legend()
    plt.show()

    return fpt_max, pdr_max


def readout_analyze(Ts, signals_g, signals_e, ro_length, plot_cum=False):
    # find the best readout window that has the largest average contrast
    contrasts = signals_g - signals_e

    # use gaussian filter to smooth the contrast
    contrasts = gaussian_filter1d(contrasts, 5)

    cum_contrasts = np.cumsum(contrasts)

    min_num = int(ro_length * len(Ts) / (Ts[-1] - Ts[0]))

    max_snr = 0
    max_idx = 0
    max_jdx = 0
    for idx in range(len(Ts) - min_num):
        jdxs = np.arange(idx + min_num, len(Ts))
        snr = np.abs(cum_contrasts[jdxs] - cum_contrasts[idx]) / np.sqrt(
            Ts[jdxs] - Ts[idx]
        )
        max_j = np.argmax(snr)
        if snr[max_j] > max_snr:
            max_snr = snr[max_j]
            max_idx = idx
            max_jdx = jdxs[max_j]

    best_offset = Ts[max_idx]
    best_length = Ts[max_jdx] - Ts[max_idx]

    # plot
    plt.figure(figsize=figsize)
    if plot_cum:
        plt.plot(Ts, np.abs(cum_contrasts), label="cum_contrast")
    else:
        plt.plot(Ts, np.abs(contrasts), label="contrast")
    plt.axvline(best_offset, color="r", ls="--", label=f"t = {best_offset:.3f}")
    plt.axvline(
        best_offset + best_length,
        color="r",
        ls="--",
        label=f"t = {best_offset + best_length:.3f}",
    )
    plt.xlabel("Time (us)")
    plt.ylabel("Magnitude (a.u.)")
    plt.legend()
    plt.show()

    return best_offset, best_length
