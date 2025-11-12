from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import scqubits as scq

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import minimize

from zcu_tools.simulate import flx2mA, mA2flx
from zcu_tools.simulate.fluxonium import calculate_eff_t1_vs_flx_with


def freq2omega(fpts: np.ndarray) -> np.ndarray:
    """GHz -> rad/ns"""
    return 2 * np.pi * fpts


def calc_therm_ratio(omega: float, T: float) -> float:
    """omega: rad/ns, T: K"""
    return (sp.constants.hbar * omega * 1e9) / (sp.constants.k * T)


def calc_Qcap_vs_omega(
    params: List[float],
    fpts: np.ndarray,
    T1s: np.ndarray,
    n_elements: np.ndarray,
    T1errs: Optional[np.ndarray] = None,
    guess_Temp: float = 20e-3,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """fpts: GHz, T1s: ns, guess_Temp: K"""
    omegas = freq2omega(fpts)

    EJ, EC, EL = params

    def spectral_density(omega) -> float:
        """omega: rad/ns, EC: GHz, T: K"""
        therm_ratio = calc_therm_ratio(omega, guess_Temp)
        s = (
            2
            * 8
            * EC
            * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
            / (1 + np.exp(-therm_ratio))
        )
        return s

    # calculate Qcap vs omega
    other_factors = spectral_density(omegas) + spectral_density(-omegas)
    Qcap_vs_omega = T1s * np.abs(n_elements) ** 2 * other_factors

    if T1errs is not None:
        Qcap_vs_omega_err = T1errs * np.abs(n_elements) ** 2 * other_factors

        return Qcap_vs_omega, Qcap_vs_omega_err
    else:
        return Qcap_vs_omega


def calc_Qind_vs_omega(
    params: List[float],
    fpts: np.ndarray,
    T1s: np.ndarray,
    phi_elements: np.ndarray,
    T1errs: Optional[np.ndarray] = None,
    guess_Temp: float = 20e-3,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """fpts: GHz, T1s: ns, guess_Temp: K"""
    omegas = freq2omega(fpts)

    EJ, EC, EL = params

    def spectral_density(omega) -> float:
        """omega: rad/ns, EL: GHz, T: K"""
        therm_ratio = calc_therm_ratio(omega, guess_Temp)
        s = (
            2
            * EL
            * (1 / np.tanh(0.5 * np.abs(therm_ratio)))
            / (1 + np.exp(-therm_ratio))
        )
        return s

    # calculate Qcap vs omega
    other_factors = spectral_density(omegas) + spectral_density(-omegas)
    Qind_vs_omega = T1s * np.abs(phi_elements) ** 2 * other_factors

    if T1errs is not None:
        Qind_vs_omega_err = T1errs * np.abs(phi_elements) ** 2 * other_factors

        return Qind_vs_omega, Qind_vs_omega_err
    else:
        return Qind_vs_omega


def find_proper_Temp(
    guess_Temp: float,
    calc_Q_fn: Callable[[float], np.ndarray],
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
    fpts: np.ndarray,
    Q_vs_omega: np.ndarray,
    Q_vs_omega_err: np.ndarray,
    Qname: str = r"$Q_{cap}$",
) -> Tuple[plt.Figure, plt.Axes]:
    """fpts: GHz, Q_vs_omega: ns/rad, Q_vs_omega_err: ns/rad"""
    omegas = freq2omega(fpts)

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
    ax: plt.Axes,
    fpts: np.ndarray,
    Q_vs_omega: np.ndarray,
    w_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    fit_constant: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """fpts: GHz, Q_vs_omega: ns/rad, w_range: (GHz, GHz)"""
    omegas = freq2omega(fpts)

    if w_range is not None:
        fit_idxs = np.arange(len(omegas))
        if w_range[0] is not None:
            fit_idxs = fit_idxs[omegas[fit_idxs] >= w_range[0]]
        if w_range[1] is not None:
            fit_idxs = fit_idxs[omegas[fit_idxs] <= w_range[1]]
        if len(fit_idxs) == 0:
            raise ValueError("No omega in the range")
        omegas = omegas[fit_idxs]
        Q_vs_omega = Q_vs_omega[fit_idxs]
    else:
        w_range = (None, None)

    if fit_constant:  # if fit_constant is True, fit a constant
        mean_Q = np.exp(np.mean(np.log(Q_vs_omega)))
        fit_Qs = np.full_like(omegas, mean_Q)

        ax.plot(omegas, fit_Qs, label=rf"$Q = {mean_Q:.1g}$")

    else:
        a, b = np.polyfit(np.log(omegas), np.log(Q_vs_omega), 1)
        Q_0, esp = np.exp(b), a

        fit_Qs = Q_0 * omegas**esp

        ax.plot(omegas, fit_Qs, label=rf"$Q(\omega) = {Q_0:.1g} \omega^{{{esp:.1f}}}$")

    ax.legend()

    return np.copy(omegas), fit_Qs


def plot_t1_vs_m01(
    elements: np.ndarray,
    T1s: np.ndarray,
    T1errs: Optional[np.ndarray] = None,
    op_name: str = r"$n_{01}$",
) -> Tuple[plt.Figure, plt.Axes]:
    """T1s: ns"""
    fig, ax = plt.subplots()

    ax.errorbar(np.abs(elements) ** 2, T1s, yerr=T1errs, fmt=".")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"|{op_name}|^2")
    ax.set_ylabel(r"$T_1$ (ns)")
    ax.set_title(r"$T_1 \; vs \; |n_{01}|^2$")
    ax.grid()

    log_product = np.log(T1s * np.abs(elements) ** 2)
    up_line_value = np.exp(np.mean(log_product) + 2 * np.std(log_product))
    down_line_value = np.exp(np.mean(log_product) - 2 * np.std(log_product))
    ax.plot(
        np.abs(elements) ** 2,
        up_line_value / np.abs(elements) ** 2,
        "k--",
        label=r"$T_1|n_{01}|^2 = $" + f"{up_line_value:.2e}",
    )
    ax.plot(
        np.abs(elements) ** 2,
        down_line_value / np.abs(elements) ** 2,
        "k--",
        label=r"$T_1|n_{01}|^2 = $" + f"{down_line_value:.2e}",
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
    """T1s: ns"""
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))

    ax.errorbar(s_mAs, s_T1s, yerr=s_T1errs, fmt=".", label="Current")

    ax.grid()
    ax.set_xlabel(r"Current (mA)", fontsize=14)
    ax.set_ylabel(r"$T_1$ (ns)", fontsize=14)
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
    fluxonium: "scq.Fluxonium",
    spectrum_data: "scq.SpectrumData",
    t_flxs: np.ndarray,
    *,
    name: str,
    noise_name: str,
    values: list[float],
    Temp: float,
    **other_noise_options: dict,
) -> Tuple[plt.Figure, plt.Axes]:
    """T1s: ns"""
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

    ax.errorbar(s_mAs, s_T1s, yerr=s_T1errs, fmt=".", label="T1")

    for i, (v, t1_eff) in enumerate(zip(values, t1_effs)):
        label = f"{name}(w)" if callable(v) else f"{name} = {v:.1e}"
        ax.plot(t_mAs, t1_eff, label=label, linestyle="--")

    range = np.ptp(s_mAs)
    ax.set_xlim(s_mAs.min() - 0.01 * range, s_mAs.max() + 0.01 * range)
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
    """T1s: ns"""
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    if title is not None:
        fig.suptitle(title)
    ax.errorbar(s_mAs, s_T1s, yerr=s_T1errs, fmt=".", label="T1")

    t_mAs = flx2mA(t_flxs, mA_c=mA_c, period=period)

    ax.plot(t_mAs, t1_effs, label=label, linestyle="--")

    range = np.ptp(s_mAs)
    ax.set_xlim(s_mAs.min() - 0.01 * range, s_mAs.max() + 0.01 * range)
    ax.set_ylim(0.5 * s_T1s.min(), 3.0 * s_T1s.max())
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
