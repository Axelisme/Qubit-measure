from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import scqubits as scq

from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flx

from .t1_curve import get_t1_vs_flx

PLOT_CUTOFF = 40
PLOT_EVALS_COUNT = 10


def plot_transitions(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    show_idxs: List[Tuple[int, int]],
    ref_freqs: Optional[List[float]] = None,
    spectrum_data: Optional[scq.SpectrumData] = None,
) -> go.Figure:
    if spectrum_data is None:
        max_level = max(j for _, j in show_idxs) + 1
        fluxonium = scq.Fluxonium(
            *params, flux=0.5, cutoff=PLOT_CUTOFF, truncated_dim=max_level
        )
        spectrum_data = fluxonium.get_spectrum_vs_paramvals(
            param_name="flux", param_vals=flxs, evals_count=max_level
        )
    evals = spectrum_data.energy_table

    fig = go.Figure()

    # plot matrix elements
    for i, j in show_idxs:
        fig.add_trace(
            go.Scatter(
                x=flxs,
                y=evals[:, j] - evals[:, i],
                mode="lines",
                name=f"{i} - {j}",
                line=dict(width=2),
            )
        )

    # plot avoid freqs
    if ref_freqs is not None:
        for freq in ref_freqs:
            fig.add_hline(y=freq, line_color="black", line_width=2, line_dash="dash")

    fig.update_layout(
        title=f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}",
        title_x=0.5,
        xaxis_title=r"$\phi_{ext}/\phi_0$",
        yaxis_title="Energy (GHz)",
    )

    return fig


def plot_matrix_elements(
    params: Tuple[float, float, float],
    flxs: np.ndarray,
    show_idxs: List[Tuple[int, int]],
    spectrum_data: Optional[scq.SpectrumData] = None,
) -> go.Figure:
    if spectrum_data is None:
        fluxonium = scq.Fluxonium(
            *params, flux=0.5, cutoff=PLOT_CUTOFF, truncated_dim=PLOT_EVALS_COUNT
        )
        spectrum_data = fluxonium.get_matelements_vs_paramvals(
            operator="n_operator",
            param_name="flux",
            param_vals=flxs,
            evals_count=PLOT_EVALS_COUNT,
        )
    matrix_elements = spectrum_data.matrixelem_table

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

    return fig


def plot_dispersive_shift(
    params: Tuple[float, float, float], flxs: np.ndarray, r_f: float, g: float
) -> go.Figure:
    fig = go.Figure()

    rf_0, rf_1 = calculate_dispersive_vs_flx(params, flxs, r_f, g)

    fig.add_hline(y=r_f, line_color="black", line_width=2, line_dash="dash")
    fig.add_trace(go.Scatter(x=flxs, y=rf_0, mode="lines", name="rf_0"))
    fig.add_trace(go.Scatter(x=flxs, y=rf_1, mode="lines", name="rf_1"))

    fig.update_layout(
        title=f"EJ/EC/EL = {params[0]:.3f}/{params[1]:.3f}/{params[2]:.3f}",
        title_x=0.5,
    )

    return fig


def plot_t1s(
    params: Tuple[float, float, float], flxs: np.ndarray, noise_channels, Temp: float
) -> go.Figure:
    fig = go.Figure()

    fluxonium = scq.Fluxonium(
        *params, flux=0.5, cutoff=PLOT_CUTOFF, truncated_dim=PLOT_EVALS_COUNT
    )
    t1s = get_t1_vs_flx(flxs, fluxonium, noise_channels=noise_channels, Temp=Temp)

    fig.add_trace(go.Scatter(x=flxs, y=t1s, mode="lines", name="t1"))
    fig.update_layout(title_x=0.51, yaxis_type="log")
    fig.update_yaxes(exponentformat="power")

    return fig
