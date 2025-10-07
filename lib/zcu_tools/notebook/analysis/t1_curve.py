from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scqubits as scq

from zcu_tools.simulate import flx2mA, mA2flx
from zcu_tools.simulate.fluxonium import calculate_eff_t1_vs_flx_with


def calc_noise_spectral(
    flxs: np.ndarray,
    T1s: np.ndarray,
    T1errs: Optional[np.ndarray] = None,
    operator: Literal["n_operator", "phi_operator"] = "n_operator",
    fluxonium: Optional[scq.Fluxonium] = None,
    spectrum_data: Optional[scq.SpectrumData] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    if spectrum_data is None:
        assert fluxonium is not None
        spectrum_data = fluxonium.get_matelements_vs_paramvals(
            operator, "flux", flxs, evals_count=20
        )
    elements = np.abs(spectrum_data.matrixelem_table[:, 0, 1])

    noise_spectrum = 1 / (T1s * elements**2)
    if T1errs is not None:
        noise_spectrum_err = T1errs / (T1s * elements) ** 2
        return noise_spectrum, noise_spectrum_err
    else:
        return noise_spectrum


def plot_t1_vs_n01_and_fit_Qcap(
    params: List[float],
    fpts: np.ndarray,
    noise_spectrum: np.ndarray,
    noise_spectrum_err: np.ndarray,
    guess_Temp: float = 20e-3,
    fit_order: int = 1,
) -> Tuple[plt.Figure, plt.Axes, List[float]]:
    omegas = 2 * np.pi * fpts * 1e-3  # MHz -> rad/ns

    EJ, EC, EL = params

    def calc_thermal_factor(omega, T):
        therm_ratio = (sp.constants.hbar * omega * 1e9) / (sp.constants.k * T)
        return (
            32
            * np.pi
            * EC
            / (np.tanh(0.5 * np.abs(therm_ratio)) * (1 + np.exp(-therm_ratio)))
        )

    # calculate spectrum without thermal factor
    thermal_factors = calc_thermal_factor(omegas, guess_Temp)
    spectrum_wo_thermal = noise_spectrum / thermal_factors
    spectrum_err_wo_thermal = noise_spectrum_err / thermal_factors

    # fit the spectrum with n order polynomial
    params = np.polyfit(omegas, np.log(spectrum_wo_thermal), deg=fit_order)

    fig, ax = plt.subplots()
    ax.set_title(r"$S(w)/\hbar^2 \; vs \; \omega$")
    ax.errorbar(omegas, spectrum_wo_thermal, yerr=spectrum_err_wo_thermal, fmt=".")
    ax.plot(
        omegas,
        np.exp(np.polyval(params, omegas)),
        label=f"fit with order {fit_order}",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$ (rad/ns)")
    ax.set_ylabel(r"$(T_1|n_{01}|^2)^{-1}$")
    ax.legend()
    ax.grid()

    return fig, ax, params


def plot_t1_vs_m01(
    elements: np.ndarray,
    T1s: np.ndarray,
    T1errs: Optional[np.ndarray] = None,
    op_name: str = r"$n_{01}$",
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    ax.errorbar(np.abs(elements) ** 2, 1e3 * T1s, yerr=1e3 * T1errs, fmt=".")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"|{op_name}|^2")
    ax.set_ylabel(r"$T_1$ (ns)")
    ax.set_title(r"$T_1 \; vs \; |n_{01}|^2$")
    ax.grid()

    T1_n01_2 = 1e3 * T1s * np.abs(elements) ** 2
    up_line_value = np.exp(np.mean(np.log(T1_n01_2)) + 2 * np.std(np.log(T1_n01_2)))
    down_line_value = np.exp(np.mean(np.log(T1_n01_2)) - 2 * np.std(np.log(T1_n01_2)))
    ax.plot(
        np.abs(elements) ** 2,
        up_line_value / np.abs(elements) ** 2,
        "k--",
        label=f"{1 / up_line_value:.2e}",
    )
    ax.plot(
        np.abs(elements) ** 2,
        down_line_value / np.abs(elements) ** 2,
        "k--",
        label=f"{1 / down_line_value:.2e}",
    )
    ax.legend()
    return fig, ax


def plot_sample_t1(
    s_mAs: np.ndarray,
    s_T1s: np.ndarray,
    s_T1errs: np.ndarray,
    mA_c: float,
    period: float,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))

    ax.errorbar(s_mAs, s_T1s, yerr=s_T1errs, fmt=".", label="Current")

    ax.grid()
    ax.set_xlabel(r"Current (mA)", fontsize=14)
    ax.set_ylabel(r"$T_1$ ($\mu s$)", fontsize=14)
    ax.set_yscale("log")

    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            partial(mA2flx, mA_c=mA_c, period=period),
            partial(flx2mA, mA_c=mA_c, period=period),
        ),
    )
    ax2.set_xlabel(r"$\phi_{ext}/\phi_0$", fontsize=14)

    return fig, ax


def plot_t1_with_sample(
    s_mAs: np.ndarray,
    s_T1s: np.ndarray,
    s_T1errs: np.ndarray,
    mA_c: float,
    period: float,
    fluxonium: scq.Fluxonium,
    spectrum_data: scq.SpectrumData,
    t_flxs: np.ndarray,
    *,
    name: str,
    noise_name: str,
    values: list[float],
    Temp: float,
    **other_noise_options: dict,
) -> Tuple[plt.Figure, plt.Axes]:
    t_mAs = flx2mA(t_flxs, mA_c=mA_c, period=period)

    t1_effs = [
        calculate_eff_t1_vs_flx_with(
            t_flxs,
            noise_channels=[(noise_name, {name: v})],
            Temp=Temp,
            fluxonium=fluxonium,
            spectrum_data=spectrum_data,
            **other_noise_options,
        )
        for v in values
    ]

    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    fig.suptitle(f"Temperature = {Temp * 1e3:.2f} mK")

    ax.errorbar(s_mAs, s_T1s * 1e3, yerr=s_T1errs * 1e3, fmt=".", label="T1")

    for i, (v, t1_eff) in enumerate(zip(values, t1_effs)):
        label = f"{name}(w)" if callable(v) else f"{name} = {v:.1e}"
        ax.plot(t_mAs, t1_eff, label=label, linestyle="--")

    ax.set_xlim(s_mAs.min() - 0.03, s_mAs.max() + 0.03)
    ax.set_ylim(0.5e3 * s_T1s.min(), 3.0e3 * s_T1s.max())
    ax.set_xlabel(r"Current (mA)", fontsize=14)
    ax.set_ylabel(r"$T_1$ (ns)", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize="x-large")
    ax.grid()

    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            partial(mA2flx, mA_c=mA_c, period=period),
            partial(flx2mA, mA_c=mA_c, period=period),
        ),
    )
    ax2.set_xlabel(r"$\phi_{ext}/\phi_0$")

    return fig, ax


def plot_eff_t1_with_sample(
    s_mAs: np.ndarray,
    s_T1s: np.ndarray,
    s_T1errs: np.ndarray,
    t1_effs: np.ndarray,
    mA_c: float,
    period: float,
    t_flxs: np.ndarray,
    *,
    label: str = "t1_eff",
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    if title is not None:
        fig.suptitle(title)
    ax.errorbar(s_mAs, s_T1s * 1e3, yerr=s_T1errs * 1e3, fmt=".-", label="T1")

    t_mAs = flx2mA(t_flxs, mA_c=mA_c, period=period)

    ax.plot(t_mAs, t1_effs, label=label, linestyle="--")

    ax.set_xlim(s_mAs.min() - 0.03, s_mAs.max() + 0.03)
    ax.set_ylim(0.5e3 * s_T1s.min(), 3.0e3 * s_T1s.max())
    ax.set_xlabel(r"Current (mA)", fontsize=14)
    ax.set_ylabel(r"$T_1$ (ns)", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize="x-large")
    ax.grid()

    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            partial(mA2flx, mA_c=mA_c, period=period),
            partial(flx2mA, mA_c=mA_c, period=period),
        ),
    )
    ax2.set_xlabel(r"$\phi_{ext}/\phi_0$")

    return fig, ax
