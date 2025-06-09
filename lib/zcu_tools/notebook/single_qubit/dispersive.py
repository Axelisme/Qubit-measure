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

    pOpt_g, err_g = fitter(fpts, g_amps)
    pOpt_e, err_e = fitter(fpts, e_amps)
    g_freq, g_kappa = pOpt_g[3], 2 * pOpt_g[4]
    e_freq, e_kappa = pOpt_e[3], 2 * pOpt_e[4]
    g_freq_err, e_freq_err = np.sqrt(np.diag(err_g))[3], np.sqrt(np.diag(err_e))[3]

    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.plot(fpts, g_amps, label="g", marker=".", c="b")
    plt.plot(fpts, e_amps, label="e", marker=".", c="r")
    plt.plot(fpts, fit_func(fpts, *pOpt_g), label=f"g fit, $kappa$={g_kappa:.1g} MHz")
    plt.plot(fpts, fit_func(fpts, *pOpt_e), label=f"e fit, $kappa$={e_kappa:.1g} MHz")
    label_g = f"$f_res$ = {g_freq:.5g} +/- {g_freq_err:.1g} MHz"
    label_e = f"$f_res$ = {e_freq:.5g} +/- {e_freq_err:.1g} MHz"
    plt.axvline(g_freq, color="b", ls="--", label=label_g)
    plt.axvline(e_freq, color="r", ls="--", label=label_e)
    plt.legend()
    plt.show()

    return abs(g_freq - e_freq) / 2, (g_kappa + e_kappa) / 2
