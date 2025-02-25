import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .general import figsize


def dispersive1D_analyze(xs, snrs, xlabel=None):
    snrs = np.abs(snrs)

    snrs = gaussian_filter1d(snrs, 1)

    max_id = np.argmax(snrs)
    max_x = xs[max_id]

    plt.figure(figsize=figsize)
    plt.plot(xs, snrs)
    plt.axvline(max_x, color="r", ls="--", label=f"max SNR = {snrs[max_id]:.2f}")
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.ylabel("SNR (a.u.)")
    plt.legend()
    plt.show()

    return max_x


def dispersive2D_analyze(xs, ys, snr2D, xlabel=None, ylabel=None):
    abssnr2D = np.abs(snr2D)

    snr2D = gaussian_filter(abssnr2D, 1)
    mask = np.isnan(snr2D)
    snr2D[mask] = abssnr2D[mask]

    x_max_id = np.nanargmax(np.nanmax(snr2D, axis=0))
    y_max_id = np.nanargmax(np.nanmax(snr2D, axis=1))
    x_max = xs[x_max_id]
    y_max = ys[y_max_id]

    plt.figure(figsize=figsize)
    plt.imshow(
        snr2D,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
    )
    plt.scatter(
        x_max, y_max, color="r", label=f"max SNR = {snr2D[y_max_id, x_max_id]:.2e}"
    )
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    plt.show()

    return x_max, y_max


def ge_lookback_analyze(Ts, signal_g, signal_e, pulse_cfg=None):
    signal_g = gaussian_filter1d(signal_g, 1)
    signal_e = gaussian_filter1d(signal_e, 1)

    amps_g = np.abs(signal_g)
    amps_e = np.abs(signal_e)
    contrast = np.abs(signal_g - signal_e)

    plt.figure(figsize=figsize)
    plt.plot(Ts, amps_g, label="g")
    plt.plot(Ts, amps_e, label="e")
    plt.plot(Ts, contrast, label="contrast")
    plt.xlabel("Time (us)")
    plt.ylabel("Magnitude (a.u.)")
    plt.legend()
    plt.show()
