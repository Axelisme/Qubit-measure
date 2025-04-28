from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scqubits as scq

from .tools import flx2mA, mA2flx


def get_eff_t1(flxs, fluxonium, noise_channels, Temp):
    plt.ioff()
    scq.settings.T1_DEFAULT_WARNING = False
    fig, ax = fluxonium.plot_t1_effective_vs_paramvals(
        param_name="flux",
        param_vals=flxs,
        xlim=([flxs.min(), flxs.max()]),
        common_noise_options=dict(i=1, j=0, T=Temp),
        noise_channels=noise_channels,
    )
    plt.close(fig)
    plt.ion()

    return ax.lines[0].get_data()[1]


def plot_eff_t1(
    s_mAs, s_flxs, s_T1s, mA_c, period, fluxonium, name, noise_name, values, Temp
):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    ax.errorbar(s_mAs, s_T1s * 1e3, fmt=".-", label="T1")
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
            s_flxs,
            fluxonium,
            noise_channels=[(noise_name, {name: v})],
            Temp=Temp,
        )
        ax.plot(s_mAs, t1_eff, label=f"{name} = {v:.1e}", linestyle="--")
        t1_effs.append(t1_eff)

    ax.set_xlim(-0.01, mA_c)
    ax.set_ylim(
        min(s_T1s.min() * 0.5e3, np.array(t1_effs).max() * 0.7),
        max(s_T1s.max() * 3.0e3, np.array(t1_effs).min() * 1.4),
    )
    ax.legend()

    return fig, ax
