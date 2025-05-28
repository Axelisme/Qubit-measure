import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .general import figsize


def dispersive1D_analyze(xs: np.ndarray, snrs: np.ndarray, xlabel: str = None) -> float:
    """
    Analyze 1D dispersive measurement data to find the maximum SNR position.

    This function processes 1D signal-to-noise ratio data, applies Gaussian smoothing,
    finds the position of maximum SNR, and visualizes the results.

    Parameters
    ----------
    xs : array-like
        X-axis values (e.g., frequency, power, etc.)
    snrs : array-like
        Signal-to-noise ratio values corresponding to xs
    xlabel : str, optional
        Label for the x-axis in the plot

    Returns
    -------
    float
        The x-value corresponding to the maximum SNR
    """
    snrs = np.abs(snrs)

    # fill NaNs with zeros
    snrs[np.isnan(snrs)] = 0

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


def dispersive_ro_len_analyze(
    ro_lens: np.ndarray, snrs: np.ndarray, t0: float = 1.0
) -> float:
    snrs = np.abs(snrs)

    # fill NaNs with zeros
    snrs[np.isnan(snrs)] = 0

    snrs = gaussian_filter1d(snrs, 1)

    # consider integral length cost
    snrs /= np.sqrt(ro_lens + t0)

    max_id = np.argmax(snrs)
    max_len = ro_lens[max_id]

    plt.figure(figsize=figsize)
    plt.plot(ro_lens, snrs)
    plt.axvline(max_len, color="r", ls="--", label=f"max SNR = {snrs[max_id]:.2f}")
    plt.xlabel("Readout length (us)")
    plt.ylabel("SNR (a.u.)")
    plt.legend()
    plt.show()

    return max_len


def dispersive2D_analyze(xs, ys, snr2D, xlabel=None, ylabel=None):
    """
    Analyze 2D dispersive measurement data to find the maximum SNR position.

    This function processes 2D signal-to-noise ratio data, applies Gaussian smoothing,
    finds the position of maximum SNR in the 2D space, and visualizes the results as a heatmap.

    Parameters
    ----------
    xs : array-like
        X-axis values (e.g., frequency, power, etc.)
    ys : array-like
        Y-axis values (e.g., time, bias, etc.)
    snr2D : 2D array-like
        2D grid of signal-to-noise ratio values
    xlabel : str, optional
        Label for the x-axis in the plot
    ylabel : str, optional
        Label for the y-axis in the plot

    Returns
    -------
    tuple
        (x_max, y_max) coordinates corresponding to the maximum SNR
    """
    snr2D = np.abs(snr2D)

    # fill NaNs with zeros
    snr2D[np.isnan(snr2D)] = 0

    snr2D = gaussian_filter(snr2D, 1)

    x_max_id = np.argmax(np.max(snr2D, axis=0))
    y_max_id = np.argmax(np.max(snr2D, axis=1))
    x_max = xs[x_max_id]
    y_max = ys[y_max_id]

    plt.figure(figsize=figsize)
    plt.imshow(
        snr2D,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
    )
    plt.scatter(
        x_max, y_max, color="r", label=f"max SNR = {snr2D[y_max_id, x_max_id]:.2e}"
    )
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.colorbar(label="SNR (a.u.)")
    plt.legend()
    plt.show()

    return x_max, y_max


def ge_lookback_analyze(Ts, signals_g, signals_e, *, pulse_cfg=None, smooth=None):
    """
    Analyze ground and excited state signals over time to evaluate measurement contrast.

    This function processes time-series data for ground and excited state signals,
    applies Gaussian smoothing, calculates the magnitude and contrast between states,
    and visualizes the results.

    Parameters
    ----------
    Ts : array-like
        Time points in microseconds
    signal_g : array-like
        Signal data for ground state
    signal_e : array-like
        Signal data for excited state
    pulse_cfg : dict, optional
        Configuration parameters for the pulse (not used in current implementation)

    Returns
    -------
    None
        This function only produces a plot but doesn't return a value
    """
    if smooth is not None:
        signals_g = gaussian_filter1d(signals_g, smooth)
        signals_e = gaussian_filter1d(signals_e, smooth)

    amps_g = np.abs(signals_g)
    amps_e = np.abs(signals_e)
    contrast = np.abs(signals_g - signals_e)

    plt.figure(figsize=figsize)
    plt.plot(Ts, amps_g, label="g")
    plt.plot(Ts, amps_e, label="e")
    plt.plot(Ts, contrast, label="contrast")

    if pulse_cfg is not None:
        trig_offset = pulse_cfg["trig_offset"]
        ro_length = pulse_cfg["ro_length"]
        plt.axvline(trig_offset, color="g", linestyle="--", label="ro start")
        plt.axvline(trig_offset + ro_length, color="g", linestyle="--", label="ro end")

    plt.xlabel("Time (us)")
    plt.ylabel("Magnitude (a.u.)")
    plt.legend()
    plt.show()
