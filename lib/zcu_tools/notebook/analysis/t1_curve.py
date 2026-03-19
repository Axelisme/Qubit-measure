from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import minimize
from typing_extensions import TYPE_CHECKING, Callable, Optional, Union

from zcu_tools.simulate import flx2value, value2flx
from zcu_tools.simulate.fluxonium import calculate_eff_t1_vs_flx_with

if TYPE_CHECKING:
    from scqubits.core.fluxonium import Fluxonium
    from scqubits.core.storage import SpectrumData


def format_exponent(n: float) -> str:
    # 將數字轉為標準科學記號字串，例如 "1.30e+03"
    a = "{:.2e}".format(n)
    base, exp = a.split("e")
    # 移除指數部分的正號與多餘的 0 (例如 +03 變成 3)
    clean_exp = int(exp)
    return rf"${base} \times 10^{{{clean_exp}}}$"


def freq2omega(freqs: NDArray[np.float64]) -> NDArray[np.float64]:
    """GHz -> rad/ns"""
    return 2 * np.pi * freqs


def charge_spectral_density(omega, Temp: float, EC: float):
    """omega: rad/ns, EC: GHz, T: K"""
    therm_ratio = calc_therm_ratio(omega, Temp)
    return (
        2
        * 8
        * EC
        * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
        / (1 + np.exp(-therm_ratio))
    )


def inductive_spectral_density(omega, Temp: float, EL: float):
    """omega: rad/ns, EL: GHz, T: K"""
    therm_ratio = calc_therm_ratio(omega, Temp)
    return (
        2 * EL * (1 / np.tanh(0.5 * np.abs(therm_ratio))) / (1 + np.exp(-therm_ratio))
    )


def calc_therm_ratio(omega, T: float):
    """omega: rad/ns, T: K"""
    return (sp.constants.hbar * omega * 1e9) / (sp.constants.k * T)


def calc_Qcap_vs_omega(
    params: tuple[float, float, float],
    freqs: NDArray[np.float64],
    T1s: NDArray[np.float64],
    n_elements: NDArray[np.float64],
    T1errs: Optional[NDArray[np.float64]] = None,
    Temp: float = 20e-3,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """fpts: GHz, T1s: ns, guess_Temp: K"""
    omegas = freq2omega(freqs)

    EJ, EC, EL = params

    # calculate Qcap vs omega
    other_factors = charge_spectral_density(omegas, Temp, EC) + charge_spectral_density(
        -omegas, Temp, EC
    )
    Qcap_vs_omega = T1s * np.abs(n_elements) ** 2 * other_factors

    if T1errs is not None:
        Qcap_vs_omega_err = T1errs * np.abs(n_elements) ** 2 * other_factors

        return Qcap_vs_omega, Qcap_vs_omega_err
    else:
        return Qcap_vs_omega


def calc_Qind_vs_omega(
    params: tuple[float, float, float],
    freqs: NDArray[np.float64],
    T1s: NDArray[np.float64],
    phi_elements: NDArray[np.float64],
    T1errs: Optional[NDArray[np.float64]] = None,
    Temp: float = 20e-3,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """fpts: GHz, T1s: ns, guess_Temp: K"""
    omegas = freq2omega(freqs)

    EJ, EC, EL = params

    # calculate Qind vs omega
    other_factors = inductive_spectral_density(
        omegas, Temp, EL
    ) + inductive_spectral_density(-omegas, Temp, EL)
    Qind_vs_omega = T1s * np.abs(phi_elements) ** 2 * other_factors

    if T1errs is not None:
        Qind_vs_omega_err = T1errs * np.abs(phi_elements) ** 2 * other_factors

        return Qind_vs_omega, Qind_vs_omega_err
    else:
        return Qind_vs_omega


def find_proper_Temp(
    guess_Temp: float,
    calc_Q_fn: Callable[[float], NDArray[np.float64]],
) -> float:
    """use scipy.optimize.minimize to find the proper Temp, the proper Temp is the one that minimizes the difference between all Q values"""

    res = minimize(
        lambda T: np.std(calc_Q_fn(T)),
        x0=[guess_Temp],
        bounds=[(10e-3, 300e-3)],
        method="L-BFGS-B",
    )

    return float(res.x[0]) if res.success else guess_Temp


def plot_Q_vs_omega(
    freqs: NDArray[np.float64],
    Q_vs_omega: NDArray[np.float64],
    Q_vs_omega_err: NDArray[np.float64],
    Qname: str = r"$Q_{cap}$",
) -> tuple[Figure, Axes]:
    """freqs: GHz, Q_vs_omega: ns/rad, Q_vs_omega_err: ns/rad"""
    omegas = freq2omega(freqs)

    fig, ax = plt.subplots()
    ax.errorbar(omegas, Q_vs_omega, yerr=Q_vs_omega_err, fmt=".", label="data")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$ (rad/ns)")
    ax.set_ylabel(Qname)
    ax.legend()
    ax.grid()

    return fig, ax


def add_Q_fit(
    ax: Axes,
    freqs: NDArray[np.float64],
    Q_vs_omega: NDArray[np.float64],
    omega_range: Optional[tuple[Optional[float], Optional[float]]] = None,
    fit_constant: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """freqs: GHz, Q_vs_omega: ns/rad, omega_range: (rad/ns, rad/ns)"""
    omegas = freq2omega(freqs)

    if omega_range is not None:
        fit_idxs = np.arange(len(omegas))
        if omega_range[0] is not None:
            fit_idxs = fit_idxs[omegas[fit_idxs] >= omega_range[0]]
        if omega_range[1] is not None:
            fit_idxs = fit_idxs[omegas[fit_idxs] <= omega_range[1]]
        if len(fit_idxs) == 0:
            raise ValueError("No omega in the range")
        omegas = omegas[fit_idxs]
        Q_vs_omega = Q_vs_omega[fit_idxs]
    else:
        omega_range = (None, None)

    if fit_constant:  # if fit_constant is True, fit a constant
        mean_Q = np.exp(np.mean(np.log(Q_vs_omega)))
        fit_Qs = np.full_like(omegas, mean_Q)

        ax.plot(omegas, fit_Qs, label=f"Q = {format_exponent(mean_Q)}")

    else:
        a, b = np.polyfit(np.log(omegas), np.log(Q_vs_omega), 1)
        Q_0, esp = np.exp(b), a

        fit_Qs = Q_0 * omegas**esp

        label = rf"$Q(\omega) = {format_exponent(Q_0)} \omega^{{{esp:.1f}}}$"
        ax.plot(omegas, fit_Qs, label=label)

    ax.legend()

    return np.copy(omegas), fit_Qs


def plot_t1_vs_m01(
    dipoles: NDArray[np.float64],
    T1s: NDArray[np.float64],
    T1errs: Optional[NDArray[np.float64]] = None,
    op_name: str = "d_{01}",
    Q_name: str = r"$Q_{cap}$",
) -> tuple[Figure, Axes]:
    """T1s: ns, Q_factors: ns/rad"""
    fig, ax = plt.subplots()

    sorted_idxs = np.argsort(dipoles)
    dipoles = dipoles[sorted_idxs]
    T1s = T1s[sorted_idxs]
    if T1errs is not None:
        T1errs = T1errs[sorted_idxs]

    ax.errorbar(dipoles, T1s, yerr=T1errs, fmt=".")
    ax.set_xscale("log")
    ax.set_yscale("log")

    log_product = np.log(T1s * dipoles)
    up_Q = np.exp(np.mean(log_product) + 2.0 * np.std(log_product))
    down_Q = np.exp(np.mean(log_product) - 2.0 * np.std(log_product))

    up_ys = up_Q / dipoles
    down_ys = down_Q / dipoles
    ax.plot(dipoles, up_ys, "k--", alpha=0.5)
    ax.plot(dipoles, down_ys, "k--", alpha=0.5)
    ax.fill_between(dipoles, up_ys, down_ys, alpha=0.05, color="k")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    angle = 5 - np.degrees(
        np.arctan2(
            np.log(xlim[1]) - np.log(xlim[0]),
            np.log(ylim[1]) - np.log(ylim[0]),
        )
    )
    text_x = np.exp(np.mean(np.log(dipoles)) + 0.5)
    idx = np.argmin(np.abs(dipoles - text_x))
    yup = up_ys[idx] * 1.01
    ydown = down_ys[idx] * 0.75

    text_kwargs = dict(rotation=angle, color="k", ha="right", va="bottom")
    ax.text(text_x, yup, f"{Q_name} = {format_exponent(up_Q)}", **text_kwargs)
    ax.text(text_x, ydown, f"{Q_name} = {format_exponent(down_Q)}", **text_kwargs)

    ax.set_xlabel(rf"$|{op_name}|^2$")
    ax.set_ylabel(r"$T_1$ (ns)")
    ax.grid()

    return fig, ax


def plot_sample_t1(
    s_dev_values: NDArray[np.float64],
    s_T1s: NDArray[np.float64],
    s_T1errs: NDArray[np.float64],
    flx_half: float,
    flx_period: float,
    xlabel: str = "Current (mA)",
) -> tuple[Figure, Axes]:
    """T1s: ns"""
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))

    s_flxs = value2flx(s_dev_values, flx_half, flx_period)

    ax.errorbar(s_flxs, s_T1s, yerr=s_T1errs, fmt=".", label="Current")

    ax.grid()
    ax.set_xlabel(r"$\phi_{ext}/\phi_0$", fontsize=14)
    ax.set_ylabel(r"$T_1$ (ns)", fontsize=14)
    ax.set_yscale("log")

    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            partial(flx2value, flx_half=flx_half, flx_period=flx_period),
            partial(value2flx, flx_half=flx_half, flx_period=flx_period),
        ),
    )
    ax2.set_xlabel(xlabel, fontsize=14)

    return fig, ax


def plot_t1_with_sample(
    s_dev_values: NDArray[np.float64],
    s_T1s: NDArray[np.float64],
    s_T1errs: NDArray[np.float64],
    flx_half: float,
    flx_period: float,
    fluxonium: Fluxonium,
    spectrum_data: SpectrumData,
    t_fluxs: NDArray[np.float64],
    *,
    name: str,
    noise_name: str,
    noise_values: list[
        Union[float, Callable[[NDArray[np.float64], float], NDArray[np.float64]]]
    ],
    Temp: float,
    xlabel: str = "Current (mA)",
    **other_noise_options: dict,
) -> tuple[Figure, Axes]:
    """T1s: ns"""
    s_flxs = value2flx(s_dev_values, flx_half, flx_period)

    t1_effs = [
        calculate_eff_t1_vs_flx_with(
            t_fluxs,
            noise_channels=[(noise_name, {name: v})],  # type: ignore
            Temp=Temp,
            fluxonium=fluxonium,
            spectrum_data=spectrum_data,
            **other_noise_options,
        )
        for v in noise_values
    ]

    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    fig.suptitle(f"Temperature = {Temp * 1e3:.2f} mK")

    ax.errorbar(s_flxs, s_T1s, yerr=s_T1errs, fmt=".", label="T1")

    nname = {
        "Q_cap": r"$Q_{cap}$",
        "x_qp": r"$x_{qp}$",
        "Q_ind": r"$Q_{ind}$",
    }[name]

    for v, t1_eff in zip(noise_values, t1_effs):
        label = f"{nname}(w)" if callable(v) else f"{nname} = {format_exponent(v)}"
        ax.plot(t_fluxs, t1_eff, label=label, linestyle="--")

    range = np.ptp(s_flxs)
    ax.set_xlim(s_flxs.min() - 0.01 * range, s_flxs.max() + 0.01 * range)
    ax.set_xlabel(r"$\phi_{ext}/\phi_0$")
    ax.set_ylabel(r"$T_1$ (ns)", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize="x-large")
    ax.grid()

    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            partial(flx2value, flx_half=flx_half, flx_period=flx_period),
            partial(value2flx, flx_half=flx_half, flx_period=flx_period),
        ),
    )
    ax2.set_xlabel(xlabel, fontsize=14)

    return fig, ax


def plot_eff_t1_with_sample(
    s_dev_values: NDArray[np.float64],
    s_T1s: NDArray[np.float64],
    s_T1errs: NDArray[np.float64],
    t1_effs: NDArray[np.float64],
    flx_half: float,
    flx_period: float,
    t_fluxs: NDArray[np.float64],
    *,
    label: str = r"$t_1^{eff}$",
    title: Optional[str] = None,
    xlabel: str = "Current (mA)",
) -> tuple[Figure, Axes]:
    """T1s: ns"""
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    if title is not None:
        fig.suptitle(title)

    s_flxs = value2flx(s_dev_values, flx_half, flx_period)
    ax.errorbar(s_flxs, s_T1s, yerr=s_T1errs, fmt=".", label="T1")

    ax.plot(t_fluxs, t1_effs, label=label, linestyle="--")

    range = np.ptp(s_flxs)
    ax.set_xlim(s_flxs.min() - 0.01 * range, s_flxs.max() + 0.01 * range)
    ax.set_ylim(0.5 * s_T1s.min(), 3.0 * s_T1s.max())
    ax.set_xlabel(r"$\phi_{ext}/\phi_0$")
    ax.set_ylabel(r"$T_1$ (ns)", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize="x-large")
    ax.grid()

    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            partial(flx2value, flx_half=flx_half, flx_period=flx_period),
            partial(value2flx, flx_half=flx_half, flx_period=flx_period),
        ),
    )
    ax2.set_xlabel(xlabel, fontsize=14)

    return fig, ax
