import matplotlib.pyplot as plt
import numpy as np

from . import fitting as ft
from .tools import convert2max_contrast

figsize = (8, 6)


def lookback_show(
    Ts, signals, plot_fit=True, ratio: float = 0.3, pulse_cfg: dict = None
):
    y = np.abs(signals)

    # find first idx where y is larger than ratio * max_y
    offset = Ts[np.argmax(y > ratio * np.max(y))]

    plt.figure(figsize=figsize)
    plt.plot(Ts, signals.real, label="I value")
    plt.plot(Ts, signals.imag, label="Q value")
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


def lookback_fft(records, xrange=(-5, 5), normalize=True, pad_ratio=1):
    def get_fft(Ts, signals):
        N = int(len(Ts) * pad_ratio)
        signals = np.pad(signals, (0, N - len(signals)), "constant")
        freqs = np.fft.fftfreq(N, (Ts[-1] - Ts[0]) / len(Ts))
        fft = np.fft.fft(signals)

        # sort the freqs and fft
        idx = np.argsort(freqs)
        freqs = freqs[idx]
        fft = fft[idx]

        if normalize:
            fft = fft / np.max(np.abs(fft))

        return freqs, fft

    results = {k: get_fft(*v) for k, v in records.items()}

    plt.figure(figsize=figsize)
    for name, (freqs, fft) in results.items():
        plt.plot(freqs, np.abs(fft), label=name)
    plt.xlim(xrange)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (a.u.)")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

    return freqs, fft


def contrast_plot(xs, signals, max_contrast=False, xlabel=None):
    fig, ax = plt.subplots(figsize=figsize)
    if max_contrast:
        y, _ = convert2max_contrast(signals.real, signals.imag)
    else:
        y = np.abs(signals)

    ax.plot(xs, y, label="signal", marker="o", markersize=3)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if max_contrast:
        ax.set_ylabel("Signal Real (a.u.)")
    else:
        ax.set_ylabel("Magnitude (a.u.)")
    ax.legend()
    plt.tight_layout()
    plt.show()


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
