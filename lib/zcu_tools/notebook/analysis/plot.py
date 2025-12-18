from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from zcu_tools.notebook.analysis.fluxdep.utils import add_secondary_xaxis
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1_vs_flx,
    calculate_energy_vs_flx,
    calculate_n_oper_vs_flx,
)
from zcu_tools.simulate.fluxonium.dispersive import calculate_chi_vs_flx

if TYPE_CHECKING:
    from scqubits.core.storage import SpectrumData

PLOT_CUTOFF = 40
PLOT_EVALS_COUNT = 15


def plot_matrix_elements(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    show_idxs: List[Tuple[int, int]],
    mAs: Optional[np.ndarray] = None,
    spectrum_data: Optional[SpectrumData] = None,
) -> Tuple[go.Figure, SpectrumData]:
    need_dim = max(max(i, j) for i, j in show_idxs) + 1

    spectrum_data, matrix_elements = calculate_n_oper_vs_flx(
        params, flxs, return_dim=need_dim, spectrum_data=spectrum_data
    )

    fig = go.Figure()
    for i, j in show_idxs:
        fig.add_trace(
            go.Scatter(
                x=flxs,
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
    if mAs is not None:
        add_secondary_xaxis(fig, flxs, mAs)

    return fig, spectrum_data


def plot_dispersive_shift(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    r_f: float,
    g: float,
    mAs: Optional[np.ndarray] = None,
    upto: int = 2,
) -> go.Figure:
    fig = go.Figure()

    chi = calculate_chi_vs_flx(params, flxs, r_f, g, res_dim=upto + 2)
    fig.add_hline(y=0.0, line_color="black", line_width=2, line_dash="dash")
    for i in range(upto):
        fig.add_trace(
            go.Scatter(
                x=flxs,
                y=(chi[:, i + 1] - chi[:, i]) * 1e3,
                mode="lines",
                name=f"chi_n{i}",
            )
        )

    fig.update_layout(
        title=f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}",
        title_x=0.5,
        xaxis_title=r"$\phi_{ext}/\phi_0$",
        yaxis_title="Chi (MHz)",
    )

    if mAs is not None:
        add_secondary_xaxis(fig, flxs, mAs)

    return fig


def plot_t1s(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    noise_channels,
    Temp: float,
    mAs: Optional[np.ndarray] = None,
) -> Tuple[go.Figure, np.ndarray]:
    fig = go.Figure()

    t1s = calculate_eff_t1_vs_flx(
        flxs,
        noise_channels=noise_channels,
        Temp=Temp,
        params=params,
        cutoff=PLOT_CUTOFF,
        evals_count=PLOT_EVALS_COUNT,
    )

    fig.add_trace(go.Scatter(x=flxs, y=t1s, mode="lines", name="t1"))
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

    if mAs is not None:
        add_secondary_xaxis(fig, flxs, mAs)

    return fig, t1s


def plot_transitions(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    show_idxs: List[Tuple[int, int]],
    ref_freqs: Optional[List[float]] = None,
) -> go.Figure:
    fig = go.Figure()

    _, energies = calculate_energy_vs_flx(params, flxs)

    for i, j in show_idxs:
        fig.add_trace(
            go.Scatter(
                x=flxs, y=energies[:, j] - energies[:, i], mode="lines", name=f"{i}-{j}"
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


def plot_mist_condition(flxs, energies, r_f, max_level=None) -> go.Figure:
    def calc_mod_transition(transition):
        return np.mod(transition + r_f / 2, r_f) - r_f / 2

    def plot_without_discontinuities(fig, x, y, **kwargs):
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
            x=flxs,
            y=np.zeros_like(flxs),
            name="0",
            line=dict(width=4, color="blue"),
        )
    )
    plot_without_discontinuities(
        fig,
        flxs,
        calc_mod_transition(energies[:, 1] - energies[:, 0]),
        name="1",
        line=dict(width=4, color="red"),
    )
    for j in range(2, max_level + 1):
        transition = energies[:, j] - energies[:, 0]
        plot_without_discontinuities(
            fig,
            flxs,
            calc_mod_transition(transition),
            line=dict(width=1, color="black", dash="dot"),
            customdata=np.floor_divide(transition, r_f),
            hovertemplate=f"state {j}, " + "n = %{customdata:.0f}<extra></extra>",
        )

    fig.update_yaxes(range=[-r_f / 2, r_f / 2])

    return fig
