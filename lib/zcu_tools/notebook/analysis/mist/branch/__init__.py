from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_chi_and_snr_over_photon(
    photons: np.ndarray,
    chi_over_n: np.ndarray,
    snrs: np.ndarray,
    qub_name: str,
    flx: float,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    best_n = photons[np.argsort(snrs)[-3]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.set_title(f"{qub_name} at flux = {flx:.1f}")
    ax1.plot(photons, np.abs(chi_over_n))
    ax1.set_ylabel(r"$|\chi_{01}|$ [GHz]")
    ax1.grid()
    ax1.set_ylim(bottom=0)

    ax2.plot(photons, snrs)
    ax2.axvline(x=best_n, color="red", linestyle="--", label=f"n = {best_n:.1f}")
    ax2.set_ylabel(r"SNR")
    ax2.set_xlabel(r"Photon number")
    ax2.grid()
    ax2.legend()
    ax2.set_ylim(bottom=0)

    return fig, (ax1, ax2)


def plot_populations_over_photon(
    branchs: List[int], photons: np.ndarray, branch_populations: Dict[int, List[float]]
) -> go.Figure:
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


def calc_critical_photons(
    photons: np.ndarray,
    populations: np.ndarray,
    critical_level: float,
) -> np.ndarray:
    critical_idx = np.argmax(populations >= critical_level, axis=-1)
    if len(critical_idx.shape) > 0:
        critical_idx[critical_idx == 0] = photons.shape[0] - 1
    else:
        critical_idx = photons.shape[0] - 1
    return photons[critical_idx]


def plot_cn_over_flx(
    flxs: np.ndarray,
    photons: np.ndarray,
    populations_over_flx: np.ndarray,
    critical_levels: Dict[int, float],
) -> go.Figure:
    # plot the critical photon number as a function of flux
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Ground State", "Excited State"),
        vertical_spacing=0.1,
    )

    for i, critical_level in enumerate(critical_levels.values()):
        pop = populations_over_flx[:, i, :]
        cn = calc_critical_photons(photons, pop, critical_level)

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


def plot_cn_with_mist(
    flxs: np.ndarray,
    photons: np.ndarray,
    populations_over_flx: np.ndarray,
    critical_levels: Dict[int, float],
    mist_flxs: np.ndarray,
    mist_photons: np.ndarray,
    mist_amps: np.ndarray,
) -> go.Figure:
    # plot the critical photon number as a function of flux
    fig = go.Figure()

    # plot the mist spectrum
    fig.add_trace(
        go.Heatmap(
            z=mist_amps.T,
            x=mist_flxs,
            y=mist_photons,
            colorscale="Viridis",
            showscale=False,
        )
    )

    flxs = np.concatenate([flxs, 1 - flxs[::-1]])
    populations_over_flx = np.concatenate(
        [populations_over_flx, populations_over_flx[::-1, ...]], axis=0
    )

    colors = ["blue", "red", "green", "yellow", "purple", "orange", "brown", "pink"]
    for b, critical_level in critical_levels.items():
        b_populations = populations_over_flx[:, b, :]

        b_populations = np.array(
            [
                np.interp(np.mod(mist_flxs, 1.0), flxs, b_population)
                for b_population in b_populations.T
            ]
        ).T

        cn = calc_critical_photons(photons, b_populations, critical_level)

        fig.add_trace(
            go.Scatter(
                x=mist_flxs,
                y=cn,
                mode="markers+lines",
                marker=dict(color=colors[b], size=6),
                line=dict(color=colors[b]),
                name=f"Branch {b}",
                # showlegend=False,
            )
        )
    fig.update_xaxes(title_text="Flux", range=[mist_flxs.min(), mist_flxs.max()])
    fig.update_yaxes(
        title_text="Photon Number",
        # range=[0.0, np.log10(mist_photons.max())],
        # type="log",
    )

    return fig
