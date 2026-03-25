from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from typing_extensions import TYPE_CHECKING, Any, Optional

from zcu_tools.notebook.analysis.fluxdep.utils import add_secondary_xaxis
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1_vs_flux,
    calculate_energy_vs_flux,
    calculate_n_oper_vs_flux,
)
from zcu_tools.simulate.fluxonium.dispersive import calculate_chi_vs_flux

if TYPE_CHECKING:
    from scqubits.core.storage import SpectrumData

PLOT_CUTOFF = 40
PLOT_EVALS_COUNT = 15


def plot_matrix_elements(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    show_idxs: list[tuple[int, int]],
    dev_values: Optional[NDArray[np.float64]] = None,
    spectrum_data: Optional[SpectrumData] = None,
) -> tuple[go.Figure, SpectrumData]:
    need_dim = max(max(i, j) for i, j in show_idxs) + 1

    spectrum_data, matrix_elements = calculate_n_oper_vs_flux(
        params, fluxs, return_dim=need_dim, spectrum_data=spectrum_data
    )

    fig = go.Figure()
    for i, j in show_idxs:
        fig.add_trace(
            go.Scatter(
                x=fluxs,
                y=np.abs(matrix_elements[:, i, j]),
                mode="lines",
                name=f"{i}-{j}",
                line=dict(width=2),
            )
        )
    fig.update_layout(
        title=f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}",
        title_x=0.5,
        xaxis_title=r"$\phi_{ext}/\phi_0$",
        yaxis_title="Matrix elements",
    )

    # Add a secondary top x-axis using provided mAs as tick labels
    if dev_values is not None:
        add_secondary_xaxis(fig, fluxs, dev_values)

    return fig, spectrum_data


def plot_dispersive_shift(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    bare_rf: float,
    g: float,
    dev_values: Optional[NDArray[np.float64]] = None,
    upto: int = 2,
) -> go.Figure:
    fig = go.Figure()

    chi = calculate_chi_vs_flux(params, fluxs, bare_rf, g, res_dim=upto + 2)

    fig.add_hline(y=0.0, line_color="black", line_width=2, line_dash="dash")

    abs_mean = 0.0
    for i in range(upto):
        diff_chi = chi[:, i + 1] - chi[:, i]
        fig.add_trace(
            go.Scatter(
                x=fluxs,
                y=diff_chi * 1e3,
                mode="lines",
                name=f"chi_n{i}",
            )
        )
        abs_mean += np.mean(np.abs(diff_chi))
    abs_mean /= upto

    fig.update_layout(
        title=f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}",
        title_x=0.5,
        xaxis_title=r"$\phi_{ext}/\phi_0$",
        yaxis_title="Chi (MHz)",
    )

    if dev_values is not None:
        add_secondary_xaxis(fig, fluxs, dev_values)

    fig.update_yaxes(range=[-abs_mean * 5e3, abs_mean * 5e3])

    return fig


def plot_t1s(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    noise_channels,
    Temp: float,
    dev_values: Optional[NDArray[np.float64]] = None,
) -> tuple[go.Figure, NDArray[np.float64]]:
    fig = go.Figure()

    t1s = calculate_eff_t1_vs_flux(
        fluxs,
        noise_channels=noise_channels,
        Temp=Temp,
        params=params,
        cutoff=PLOT_CUTOFF,
        evals_count=PLOT_EVALS_COUNT,
    )

    fig.add_trace(go.Scatter(x=fluxs, y=t1s, mode="lines", name="t1"))
    fig.update_layout(
        title=dict(
            text=f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}",
            x=0.51,
        ),
        xaxis_title=r"$\phi_{ext}/\phi_0$",
        yaxis_title="T1 (ns)",
        yaxis_type="log",
    )
    fig.update_yaxes(exponentformat="power")

    if dev_values is not None:
        add_secondary_xaxis(fig, fluxs, dev_values)

    return fig, t1s


def plot_transitions(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    show_idxs: list[tuple[int, int]],
    ref_freqs: Optional[list[float]] = None,
) -> go.Figure:
    fig = go.Figure()

    _, energies = calculate_energy_vs_flux(params, fluxs)

    for i, j in show_idxs:
        fig.add_trace(
            go.Scatter(
                x=fluxs,
                y=energies[:, j] - energies[:, i],
                mode="lines",
                name=f"{i}-{j}",
            )
        )
    if ref_freqs is not None:
        for ref_freq in ref_freqs:
            fig.add_hline(
                y=ref_freq, line_color="black", line_width=2, line_dash="dash"
            )
    fig.update_layout(
        title=f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}",
        title_x=0.51,
    )

    return fig


def plot_mist_condition(
    fluxs: NDArray[np.float64],
    energies: NDArray[np.float64],
    r_f: float,
    max_level: Optional[int] = None,
) -> go.Figure:
    def calc_mod_transition(transition: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.mod(transition + r_f / 2, r_f) - r_f / 2

    def plot_without_discontinuities(
        fig: go.Figure,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        **kwargs: Any,
    ) -> None:
        discontinuities = np.where(np.abs(np.diff(y)) > r_f / 2)[0]
        x_new = np.insert(x, discontinuities + 1, np.nan)
        y_new = np.insert(y, discontinuities + 1, np.nan)
        kwargs.setdefault("mode", "lines")
        fig.add_trace(go.Scatter(x=x_new, y=y_new, **kwargs))

    if max_level is None:
        max_level = energies.shape[1] - 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fluxs,
            y=np.zeros_like(fluxs),
            name="0",
            line=dict(width=4, color="blue"),
        )
    )
    plot_without_discontinuities(
        fig,
        fluxs,
        calc_mod_transition(energies[:, 1] - energies[:, 0]),
        name="1",
        line=dict(width=4, color="red"),
    )
    for j in range(2, max_level + 1):
        transition = energies[:, j] - energies[:, 0]
        plot_without_discontinuities(
            fig,
            fluxs,
            calc_mod_transition(transition),
            line=dict(width=1, color="black", dash="dot"),
            customdata=np.floor_divide(transition, r_f),
            hovertemplate=f"state {j}, " + "n = %{customdata:.0f}<extra></extra>",
        )

    fig.update_yaxes(range=[-r_f / 2, r_f / 2])

    return fig
