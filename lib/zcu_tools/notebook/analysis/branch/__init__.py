from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots

from . import floquet, full_quantum


def plot_populations_over_photon(
    branchs: List[int], photons: np.ndarray, branch_populations: Dict[int, List[float]]
) -> Figure:
    fig = go.Figure()

    for b in branchs:
        pop_b = branch_populations[b]
        if np.ptp(pop_b) > 1.0:
            color = None
            name = f"Branch {b}"
            showlegend = True
        else:
            color = "lightgrey"
            name = f"Branch {b}"
            showlegend = False

        fig.add_trace(
            go.Scatter(
                x=photons,
                y=pop_b,
                mode="lines",
                name=name,
                line=dict(color=color),
                showlegend=showlegend,
            )
        )

    fig.update_layout(
        title="Branch Populations",
        title_x=0.51,
        xaxis_title="Photons",
        yaxis_title="Population",
        showlegend=True,
    )

    return fig


def plot_cn_over_flx(
    flxs: np.ndarray,
    photons: np.ndarray,
    populations_over_flx: np.ndarray,
    critical_levels: Dict[int, float],
) -> Figure:
    # plot the critical photon number as a function of flux
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Ground State", "Excited State"),
        vertical_spacing=0.1,
    )

    # g_populations = populations_over_flx[:, 0, :]
    # e_populations = populations_over_flx[:, 1, :]
    for i, (b, critical_level) in enumerate(critical_levels.items()):
        pop = populations_over_flx[:, i, :]
        cn = np.argmax(pop >= critical_level, axis=1)
        cn[cn == 0] = pop.shape[1] - 1
        cn = photons[cn]

        fig.add_trace(
            go.Heatmap(
                z=pop.T,
                x=flxs,
                y=photons,
                colorscale="Viridis",
                zmin=0,
                zmax=critical_level,
                showscale=False,
            ),
            row=i + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=flxs,
                y=cn,
                mode="markers+lines",
                marker=dict(color="red", size=6),
                line=dict(color="red"),
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )
    fig.update_xaxes(title_text="Flux", row=2, col=1)
    fig.update_yaxes(title_text="Photon Number", row=1, col=1)
    fig.update_yaxes(title_text="Photon Number", row=2, col=1)

    return fig
