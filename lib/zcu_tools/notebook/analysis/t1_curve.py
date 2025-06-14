from functools import partial
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scqubits as scq

from zcu_tools.simulate import flx2mA, mA2flx


def get_eff_t1(
    flx: float, fluxonium: scq.Fluxonium, noise_channels: list, Temp: float, esys=None
) -> float:
    scq.settings.T1_DEFAULT_WARNING = False
    fluxonium.flux = flx
    kwargs = {
        "noise_channels": noise_channels,
        "common_noise_options": dict(i=1, j=0, T=Temp),
    }
    if esys is not None:
        kwargs["esys"] = esys
    return fluxonium.t1_effective(**kwargs)


def get_t1_vs_flx(
    flxs: np.ndarray,
    fluxonium: scq.Fluxonium,
    noise_channels: list,
    Temp: float | np.ndarray,
    esys: Optional[np.ndarray] = None,
    flx_range: Optional[tuple] = None,
) -> np.ndarray:
    scq.settings.T1_DEFAULT_WARNING = False
    t1_effs = []
    for i, flx in enumerate(flxs):
        if flx_range is not None:
            if flx < flx_range[0] or flx > flx_range[1]:
                t1_effs.append(np.nan)
                continue

        temp = Temp if isinstance(Temp, (int, float)) else Temp[i]
        esys_i = (esys[0][i], esys[1][i]) if esys is not None else None
        t1 = get_eff_t1(flx, fluxonium, noise_channels, temp, esys_i)
        t1_effs.append(t1)
    return np.array(t1_effs)


def plot_t1_vs_flx(
    s_mAs,
    s_flxs,
    s_T1s,
    s_T1errs,
    mA_c,
    period,
    fluxonium,
    name,
    noise_name,
    values,
    Temp,
    t_mAs=None,
    t_flxs=None,
    esys=None,
) -> Tuple[plt.Figure, plt.Axes]:
    if t_mAs is None:
        t_mAs = s_mAs
    if t_flxs is None:
        t_flxs = s_flxs

    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    fig.suptitle(f"Temperature = {Temp * 1e3} mK")
    ax.errorbar(s_mAs, s_T1s * 1e3, yerr=s_T1errs * 1e3, fmt=".-", label="T1")
    ax.set_xlabel(r"Current (mA)", fontsize=14)
    ax.set_ylabel(r"$T_1$ (ns)", fontsize=14)
    ax.set_yscale("log")
    ax.grid()
    ax2 = ax.secondary_xaxis(
        "top",
        functions=(
            partial(mA2flx, mA_c=mA_c, period=period),
            partial(flx2mA, mA_c=mA_c, period=period),
        ),
    )
    ax2.set_xlabel(r"$\phi_{ext}/\phi_0$")

    t1_effs = []
    for v in values:
        t1_eff = get_t1_vs_flx(
            t_flxs,
            fluxonium,
            noise_channels=[(noise_name, {name: v})],
            Temp=Temp,
            esys=esys,
            flx_range=(s_flxs.min() - 0.1, s_flxs.max() + 0.1),
        )
        ax.plot(t_mAs, t1_eff, label=f"{name} = {v:.1e}", linestyle="--")
        t1_effs.append(t1_eff)

    ax.set_xlim(s_mAs.min() - 0.03, s_mAs.max() + 0.03)
    # ax.set_ylim(
    #     min(s_T1s.min() * 0.5e3, np.array(t1_effs).max() * 0.7),
    #     max(s_T1s.max() * 3.0e3, np.array(t1_effs).min() * 1.4),
    # )
    ax.legend(fontsize="x-large")

    return fig, ax


def plot_sample_t1(
    s_mAs: np.ndarray,
    s_T1s: np.ndarray,
    s_T1errs: np.ndarray,
    mA_c: float,
    period: float,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax1 = plt.subplots(constrained_layout=True, figsize=(8, 4))

    ax1.errorbar(s_mAs, s_T1s, yerr=s_T1errs, fmt=".-", label="Current")
    ax1.grid()
    ax1.set_xlabel(r"Current (mA)", fontsize=14)
    ax1.set_ylabel(r"$T_1$ ($\mu s$)", fontsize=14)
    ax2 = ax1.secondary_xaxis(
        "top",
        functions=(
            partial(mA2flx, mA_c=mA_c, period=period),
            partial(flx2mA, mA_c=mA_c, period=period),
        ),
    )
    ax2.set_xlabel(r"$\phi_{ext}/\phi_0$", fontsize=14)

    return fig, ax1
