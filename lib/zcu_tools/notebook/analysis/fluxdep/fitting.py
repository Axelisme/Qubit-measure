"""Notebook flux-dependent search entry and spectrum refinement."""

from __future__ import annotations

from io import BytesIO
from typing import Literal, overload

import numpy as np
from IPython.display import Image, display
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import least_squares
from tqdm.auto import tqdm

from zcu_tools.analysis.fluxdep.models import TransitionDict, energy2linearform
from zcu_tools.analysis.fluxdep.search import (
    ParamBounds,
    SearchExecution,
    search_database,
)
from zcu_tools.analysis.fluxdep.search_models import count_max_evals
from zcu_tools.plotting.fluxdep import make_search_diagnostic_figure
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux


@overload
def search_in_database(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    datapath: str,
    transitions: TransitionDict,
    EJb: tuple[float, float],
    ECb: tuple[float, float],
    ELb: tuple[float, float],
    *,
    n_jobs: int = 1,
    plot: Literal[True] = True,
) -> tuple[tuple[float, float, float], Figure]: ...


@overload
def search_in_database(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    datapath: str,
    transitions: TransitionDict,
    EJb: tuple[float, float],
    ECb: tuple[float, float],
    ELb: tuple[float, float],
    *,
    n_jobs: int = 1,
    plot: Literal[False],
) -> tuple[tuple[float, float, float], None]: ...


def search_in_database(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    datapath: str,
    transitions: TransitionDict,
    EJb: tuple[float, float],
    ECb: tuple[float, float],
    ELb: tuple[float, float],
    *,
    n_jobs: int = 1,
    plot: bool = True,
) -> tuple[tuple[float, float, float], Figure | None]:
    result = search_database(
        fluxs,
        freqs,
        datapath,
        transitions,
        ParamBounds(EJ=EJb, EC=ECb, EL=ELb),
        execution=SearchExecution(n_jobs=n_jobs),
    )
    fig: Figure | None = None
    if plot:
        fig = make_search_diagnostic_figure(result)
        image = BytesIO()
        fig.savefig(image, format="png")
        display(Image(data=image.getvalue()))
    return result.params, fig


def fit_spectrum(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    init_params: tuple[float, float, float],
    transitions: TransitionDict,
    param_b: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    maxfun: int = 1000,
) -> tuple[float, float, float]:
    max_lvl = count_max_evals(transitions)

    pbar = tqdm(desc="Distance: nan", total=maxfun, leave=False)

    def update_pbar(params, dist) -> None:
        nonlocal pbar

        pbar.set_postfix_str(f"({params[0]:.3f}, {params[1]:.3f}, {params[2]:.2f})")
        pbar.set_description_str(f"Distance: {dist:.2g}")
        pbar.update()

    # 使用 least_squares 進行參數最佳化
    def residuals(params) -> np.ndarray:
        nonlocal fluxs, transitions, freqs

        # 計算能量並轉成線性形式
        _, energies = calculate_energy_vs_flux(
            params, fluxs, cutoff=45, evals_count=max_lvl
        )
        Bs, Cs = energy2linearform(energies, transitions)
        # 計算每個點的最小誤差
        dists = np.min(np.abs(freqs[:, None] - np.abs(Bs + Cs)), axis=1)

        update_pbar(params, np.mean(dists))

        return dists

    import scqubits.settings as scq_settings

    old = scq_settings.PROGRESSBAR_DISABLED
    scq_settings.PROGRESSBAR_DISABLED = True

    EJb, ECb, ELb = param_b
    try:
        res = least_squares(
            residuals,
            init_params,
            bounds=((EJb[0], ECb[0], ELb[0]), (EJb[1], ECb[1], ELb[1])),
            max_nfev=maxfun,
            loss="soft_l1",
        )
    finally:
        scq_settings.PROGRESSBAR_DISABLED = old
        pbar.close()

    best_params = res.x

    return (float(best_params[0]), float(best_params[1]), float(best_params[2]))
