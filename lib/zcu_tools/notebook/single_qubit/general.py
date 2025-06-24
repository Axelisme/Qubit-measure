from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

import zcu_tools.notebook.util.fitting as ft

from .process import rotate2real

figsize = (8, 6)


def lookback_show(
    Ts,
    signals,
    *,
    plot_fit=True,
    smooth=None,
    ratio: float = 0.3,
    ro_cfg: Optional[dict] = None,
) -> float:
    if smooth is not None:
        signals = gaussian_filter1d(signals, smooth)
    y = np.abs(signals)

    # find first idx where y is larger than ratio * max_y
    offset = Ts[np.argmax(y > ratio * np.max(y))]

    plt.figure(figsize=figsize)
    plt.plot(Ts, signals.real, label="I value")
    plt.plot(Ts, signals.imag, label="Q value")
    plt.plot(Ts, y, label="mag")
    if plot_fit:
        plt.axvline(offset, color="r", linestyle="--", label="predict_offset")
    if ro_cfg is not None:
        trig_offset = ro_cfg["trig_offset"]
        ro_length = ro_cfg["ro_length"]
        plt.axvline(trig_offset, color="g", linestyle="--", label="ro start")
        plt.axvline(trig_offset + ro_length, color="g", linestyle="--", label="ro end")

    plt.xlabel("Time (us)")
    plt.ylabel("a.u.")
    plt.legend()
    plt.show()

    return offset


def lookback_fft(
    records: Dict[str, Tuple[np.ndarray, np.ndarray]],
    xrange=(-5, 5),
    normalize=True,
    pad_ratio=1,
) -> Tuple[np.ndarray, np.ndarray]:
    def get_fft(Ts: np.ndarray, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def contrast_plot(
    xs: np.ndarray, signals: np.ndarray, max_contrast=False, xlabel=None
) -> None:
    if max_contrast:
        y = rotate2real(signals).real
    else:
        y = np.abs(signals)

    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout(pad=3)
    ax.plot(xs, y, label="signal", marker="o", markersize=3)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if max_contrast:
        ax.set_ylabel("Signal Real (a.u.)")
    else:
        ax.set_ylabel("Magnitude (a.u.)")
    ax.legend()
    plt.show()


def phase_analyze(
    fpts: np.ndarray, signals: np.ndarray, plot=True, plot_fit=True
) -> Tuple[float, float]:
    phase = np.angle(signals)
    phase = np.unwrap(phase) * 180 / np.pi

    pOpt, _ = ft.fit_line(fpts, phase)
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


def freq_analyze(
    fpts: np.ndarray,
    signals: np.ndarray,
    type: Literal["lor", "sinc"] = "lor",
    asym=False,
    plot_fit=True,
    max_contrast=False,
) -> Tuple[float, float]:
    val_mask = ~np.isnan(signals)
    fpts = fpts[val_mask]
    signals = signals[val_mask]

    if max_contrast:
        y = rotate2real(signals).real
    else:
        y = np.abs(signals)

    if type == "lor":
        if asym:
            pOpt, err = ft.fit_asym_lor(fpts, y)
            curve = ft.asym_lorfunc(fpts, *pOpt)
        else:
            pOpt, err = ft.fitlor(fpts, y)
            curve = ft.lorfunc(fpts, *pOpt)
        freq: float = pOpt[3]
        kappa: float = 2 * pOpt[4]
        freq_err = np.sqrt(np.diag(err))[3]
    elif type == "sinc":
        if asym:
            raise ValueError("Asymmetric sinc fit is not supported.")
        pOpt, err = ft.fitsinc(fpts, y)
        curve = ft.sincfunc(fpts, *pOpt)
        freq: float = pOpt[3]
        kappa: float = 1.2067 * pOpt[4]  # sinc function hwm is 1.2067 * gamma
        freq_err = np.sqrt(np.diag(err))[3]

    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.plot(fpts, y, label="signal", marker="o", markersize=3)
    if plot_fit:
        plt.plot(fpts, curve, label=f"fit, $kappa$={kappa:.1g} MHz")
        label = f"$f_res$ = {freq:.5g} +/- {freq_err:.1g} MHz"
        plt.axvline(freq, color="r", ls="--", label=label)
    plt.xlabel("Frequency (MHz)")
    if max_contrast:
        plt.ylabel("Signal Real")
    else:
        plt.ylabel("Magnitude")
    plt.legend()
    plt.show()

    return freq, kappa


def effective_temperature(
    population: List[Tuple[float, float]], plot=True
) -> Tuple[float, float]:
    """
    Calculate the effective temperature of a population of qubits.

    Parameters
    ----------
    population : List[Tuple[float, float]]
        A list of tuples of (population, energy in MHz).
    plot : bool, default=True
        Whether to plot the population and energy.

    Returns
    -------
    Tuple[float, float]
        The effective temperature in mK and its error.
    """

    def boltzmann_distribution(freq: float, eff_T: float) -> float:
        exp_term = np.exp(-1e6 * sc.h * freq / (sc.k * 1e-3 * eff_T))
        return exp_term / np.sum(exp_term)

    # calculate the effective temperature
    if len(population) < 2:
        raise ValueError(
            "At least two qubits are required to calculate effective temperature."
        )

    pops, freqs = zip(*population)
    pops, freqs = np.array(pops), np.array(freqs)

    # directly calculate from two points
    eff_T = 1e9 * sc.h * (freqs[1] - freqs[0]) / (sc.k * np.log(pops[0] / pops[1]))
    err_T = 0.0
    if len(population) > 2:
        # fit the boltzmann distribution
        pOpt, err = curve_fit(boltzmann_distribution, freqs, pops, p0=(eff_T,))
        eff_T = pOpt[0]
        err_T = np.sqrt(np.diag(err))[0]

    if plot:
        plt.figure(figsize=figsize)
        plt.plot(freqs, pops, label="data")
        plt.plot(
            freqs,
            boltzmann_distribution(freqs, eff_T),
            label=f"fit, $T_e$={eff_T:.1g} +/- {err_T:.1g} K",
        )
        plt.xlabel("Energy (a.u.)")
        plt.ylabel("Population")
        plt.legend()
        plt.show()

    return eff_T, err_T
