import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from zcu_tools.notebook.single_qubit.process import rotate2real

from .general import figsize


def mux_reset_fpt_analyze(fpts1, fpts2, signals, xname=None, yname=None, smooth=1):
    signals = gaussian_filter(signals, smooth)

    amps = np.abs(signals - np.mean(signals))
    x_peak = fpts1[np.argmax(np.max(amps, axis=1))]
    y_peak = fpts2[np.argmax(np.max(amps, axis=0))]

    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()
    ax.imshow(
        rotate2real(signals.T).real,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=(fpts1[0], fpts1[-1], fpts2[0], fpts2[-1]),
    )
    peak_label = f"({x_peak:.1f}, {y_peak:.1f}) MHz"
    ax.scatter(x_peak, y_peak, color="r", s=40, marker="*", label=peak_label)
    if xname is not None:
        plt.xlabel(f"{xname} Frequency (MHz)", fontsize=14)
    if yname is not None:
        plt.ylabel(f"{yname} Frequency (MHz)", fontsize=14)
    ax.legend(fontsize="x-large")
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.show()

    return x_peak, y_peak


def mux_reset_pdr_analyze(pdrs1, pdrs2, signal2D, xname=None, yname=None, smooth=1):
    signal2D = gaussian_filter(signal2D, smooth)

    amp2D = rotate2real(signal2D).real

    if amp2D[0, 0] < np.mean(amp2D):
        x_id = np.argmax(np.max(amp2D, axis=1))
        y_id = np.argmax(np.max(amp2D, axis=0))
    else:
        x_id = np.argmin(np.min(amp2D, axis=1))
        y_id = np.argmin(np.min(amp2D, axis=0))
        amp2D = np.mean(amp2D) - amp2D
    x_peak = pdrs1[x_id]
    y_peak = pdrs2[y_id]

    plt.figure(figsize=figsize)
    plt.imshow(
        amp2D.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=(pdrs1[0], pdrs1[-1], pdrs2[0], pdrs2[-1]),
    )
    peak_label = f"({x_peak:.1f}, {y_peak:.1f}) a.u."
    plt.scatter(
        x_peak,
        y_peak,
        color="r",
        s=40,
        marker="*",
        label=peak_label,
    )
    if xname is not None:
        plt.xlabel(f"{xname} gain (a.u.)", fontsize=14)
    if yname is not None:
        plt.ylabel(f"{yname} gain (a.u.)", fontsize=14)
    plt.legend(fontsize="x-large")
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.show()

    return x_peak, y_peak


def mux_reset_time_analyze(Ts, signals) -> None:
    signals = rotate2real(signals).real

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(Ts, signals, marker=".")
    ax.set_xlabel("ProbeTime (us)", fontsize=14)
    ax.set_ylabel("Signal (a.u.)", fontsize=14)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.show()
