from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import zcu_tools.notebook.util.fitting as ft

from .general import figsize


def analyze_dispersive(
    fpts: np.ndarray, signals: np.ndarray, asym=True
) -> Tuple[float, float]:
    amps = np.abs(signals)
    g_amps, e_amps = amps[0, :], amps[1, :]

    fitter = ft.fit_asym_lor if asym else ft.fitlor
    fit_func = ft.asym_lorfunc if asym else ft.lorfunc

    pOpt_g, _ = fitter(fpts, g_amps)
    pOpt_e, _ = fitter(fpts, e_amps)
    g_freq, g_kappa = pOpt_g[3], 2 * pOpt_g[4]
    e_freq, e_kappa = pOpt_e[3], 2 * pOpt_e[4]

    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.plot(fpts, g_amps, marker=".", c="b")
    plt.plot(fpts, e_amps, marker=".", c="r")
    plt.plot(fpts, fit_func(fpts, *pOpt_g))
    plt.plot(fpts, fit_func(fpts, *pOpt_e))
    label_g = f"ground : {g_freq:.1f}MHz, $kappa$={g_kappa:.1f} MHz"
    label_e = f"excited : {e_freq:.1f}MHz, $kappa$={e_kappa:.1f} MHz"
    plt.axvline(g_freq, color="b", ls="--", label=label_g)
    plt.axvline(e_freq, color="r", ls="--", label=label_e)
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.show()

    return abs(g_freq - e_freq) / 2, (g_kappa + e_kappa) / 2
