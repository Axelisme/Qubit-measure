import numpy as np
import plotly.graph_objects as go
import qutip as qt
from zcu_tools.simulate.fluxonium.branch.floquet import FloquetBranchAnalysis


def calc_overlay(
    params, photons, r_f: float, g: float, flx: float, qub_dim=30, qub_cutoff=40
) -> np.ndarray:
    f_analysis = FloquetBranchAnalysis(
        params, r_f, g, flx, qub_dim=qub_dim, qub_cutoff=qub_cutoff
    )

    def calc_max_overlay(states, target_state):
        return np.max([np.abs(state.dag() @ target_state) for state in states])

    overlays = np.zeros((len(photons), 2), dtype=np.float32)
    for i, n in enumerate(photons):
        # calculate time average of states
        states_n = f_analysis.make_floquet_basis(photon=n).state(t=0)

        # calculate critical photon number for ground and excited state
        overlays[i, 0] = calc_max_overlay(states_n, qt.basis(qub_dim, 0))
        overlays[i, 1] = calc_max_overlay(states_n, qt.basis(qub_dim, 1))

    return overlays


def plot_overlay(
    fig, name, overlay, sim_flxs, sim_photons, threshold, row=1, col=1, line_kwargs=None
) -> None:
    crit_idxs = np.argmax(overlay < threshold, axis=1)
    crit_idxs[crit_idxs == 0] = len(sim_photons) - 1
    crit_ns = sim_photons[crit_idxs]

    if line_kwargs is None:
        line_kwargs = dict()
    line_kwargs.setdefault("width", 1)

    fig.add_trace(
        go.Heatmap(
            z=overlay.T, x=sim_flxs, y=sim_photons, colorscale="Greys", showscale=False
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=sim_flxs,
            y=crit_ns,
            mode="lines",
            line=line_kwargs,
            name=name,
            showlegend=True,
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(range=[sim_photons[0], sim_photons[-1]], row=row, col=col)
