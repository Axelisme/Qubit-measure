from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scqubits as scq

from .tools import flx2mA, mA2flx


def get_eff_t1(
    flxs: np.ndarray,
    fluxonium: scq.Fluxonium,
    noise_channels: list,
    Temp: float | np.ndarray,
    esys: Optional[np.ndarray] = None,
    flx_range: Optional[tuple] = None,
):
    scq.settings.T1_DEFAULT_WARNING = False
    t1_effs = []
    for i, flx in enumerate(flxs):
        if flx_range is not None:
            if flx < flx_range[0] or flx > flx_range[1]:
                t1_effs.append(np.nan)
                continue

        fluxonium.flux = flx
        temp = Temp if isinstance(Temp, (int, float)) else Temp[i]
        kwargs = {
            "noise_channels": noise_channels,
            "common_noise_options": dict(i=1, j=0, T=temp),
        }
        if esys is not None:
            kwargs["esys"] = (esys[0][i], esys[1][i])

        t1_effs.append(fluxonium.t1_effective(**kwargs))
    return np.array(t1_effs)


def plot_eff_t1(
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
):
    if t_mAs is None:
        t_mAs = s_mAs
    if t_flxs is None:
        t_flxs = s_flxs

    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    fig.suptitle(f"Temperature = {Temp * 1e3} mK")
    ax.errorbar(s_mAs, s_T1s * 1e3, yerr=s_T1errs * 1e3, fmt=".-", label="T1")
    ax.set_xlabel(r"Current (mA)")
    ax.set_ylabel(r"$T_1$ (ns)")
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
        t1_eff = get_eff_t1(
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
    ax.set_ylim(
        min(s_T1s.min() * 0.5e3, np.array(t1_effs).max() * 0.7),
        max(s_T1s.max() * 3.0e3, np.array(t1_effs).min() * 1.4),
    )
    ax.legend()

    return fig, ax
