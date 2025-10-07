from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scqubits as scq
from scipy.optimize import least_squares

from zcu_tools.simulate import flx2mA, mA2flx
from zcu_tools.simulate.fluxonium import calculate_eff_t1_vs_flx_with


def calc_noise_spectral(
    flxs: np.ndarray,
    T1s: np.ndarray,
    fluxonium: scq.Fluxonium,
    T1errs: Optional[np.ndarray] = None,
    operator: Literal["n_operator", "phi_operator"] = "n_operator",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    elements = np.abs(
        fluxonium.get_matelements_vs_paramvals(
            operator, "flux", flxs, evals_count=20
        ).matrixelem_table[:, 0, 1]
    )

    noise_spectrum = 1 / (T1s * elements**2)
    if T1errs is not None:
        noise_spectrum_err = T1errs / (T1s * elements) ** 2
        return noise_spectrum, noise_spectrum_err
    else:
        return noise_spectrum


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

    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            partial(mA2flx, mA_c=mA_c, period=period),
            partial(flx2mA, mA_c=mA_c, period=period),
        ),
    )
    ax2.set_xlabel(r"$\phi_{ext}/\phi_0$", fontsize=14)

    return fig, ax


def fit_noise_and_temp(
    s_flxs: np.ndarray,
    s_T1s: np.ndarray,
    fluxonium: scq.Fluxonium,
    init_guess_noise: List[Tuple[str, Dict[str, float]]],
    bounds_noise: List[Tuple[str, Dict[str, Optional[Tuple[float, float]]]]],
    init_guess_temp: float,
    bounds_temp: Optional[Tuple[float, float]] = None,
    evals_count: int = 20,
    asym_loss_weight: float = 1.0,
) -> Tuple[List[Tuple[str, Dict[str, float]]], float]:
    spectrum_data = fluxonium.get_spectrum_vs_paramvals(
        "flux", s_flxs, evals_count=evals_count, get_eigenstates=True
    )

    # Flatten initial guesses and construct bounds vectors
    # initial values for free parameters
    param_init: List[float] = []
    # mapping of free param index -> (channel_idx, param_name)
    param_meta: List[Tuple[int, str]] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    # Build fast look-up for (low, high) bounds for each channel/parameter.
    bounds_dicts = {idx: params for idx, (_, params) in enumerate(bounds_noise)}

    for ch_idx, (_, param_dict) in enumerate(init_guess_noise):
        bdict = bounds_dicts.get(ch_idx, {})
        for p_name, p_val in param_dict.items():
            # bounds may be None -> fixed parameter
            bound_pair = bdict.get(p_name, (0.0, np.inf))
            if bound_pair is None:
                # leave value fixed, not part of optimization vector
                continue
            low, high = bound_pair
            param_init.append(p_val)
            param_meta.append((ch_idx, p_name))
            lower_bounds.append(low)
            upper_bounds.append(high)

    # Append temperature as the last parameter and its bounds
    temp_is_variable = bounds_temp is not None
    if temp_is_variable:
        low_temp, high_temp = bounds_temp
        param_init.append(init_guess_temp)
        lower_bounds.append(low_temp)
        upper_bounds.append(high_temp)

    def vector_to_noise_channels(
        x: np.ndarray,
    ) -> Tuple[List[Tuple[str, Dict[str, float]]], float]:
        """Convert flat parameter vector back to noise_channels list and Temp."""
        # Deep-copy the channel structure with updated parameter values
        noise_channels = [
            (ch_name, {k: v for k, v in params.items()})
            for ch_name, params in init_guess_noise
        ]

        x_iter = iter(x)

        # Fill in optimized noise parameters
        for idx, (ch_idx, p_name) in enumerate(param_meta):
            noise_channels[ch_idx][1][p_name] = next(x_iter)

        if temp_is_variable:
            Temp = float(next(x_iter))
        else:
            Temp = init_guess_temp
        return noise_channels, Temp

    # Residual function in log-space to equally weight different magnitudes
    def residuals(x: np.ndarray) -> np.ndarray:
        noise_channels, Temp = vector_to_noise_channels(x)
        t1_theory = calculate_eff_t1_vs_flx_with(
            s_flxs,
            noise_channels,
            Temp,
            fluxonium=fluxonium,
            spectrum_data=spectrum_data,
        )
        # Avoid log of zero or negative by flooring at a tiny value
        t1_theory = np.clip(t1_theory, 1e-20, None)
        diff = np.log10(t1_theory) - np.log10(s_T1s)

        # Asymmetric weighting: penalize diff < 0 by factor 10 (sqrt factor for residuals)
        weight = np.where(diff < 0, np.sqrt(asym_loss_weight), 1.0)
        return weight * diff

    if len(param_init) == 0:
        # No parameters free -> return initial guess directly
        return init_guess_noise, init_guess_temp

    result = least_squares(
        residuals,
        x0=np.asarray(param_init, dtype=float),
        bounds=(lower_bounds, upper_bounds),
        method="trf",
    )

    fitted_noise_channels, fitted_Temp = vector_to_noise_channels(result.x)

    return fitted_noise_channels, fitted_Temp


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
        label = f"{name}(w)_{i}" if callable(v) else f"{name} = {v:.1e}"
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
