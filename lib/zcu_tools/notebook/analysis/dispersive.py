from functools import lru_cache
from typing import Optional, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from scipy.optimize import minimize
from tqdm.auto import tqdm

from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flx


def search_proper_g(
    params: Tuple[float, float, float],
    r_f: float,
    sp_flxs: np.ndarray,
    sp_fpts: np.ndarray,
    signals: np.ndarray,
    g_bound: Tuple[float, float],
    g_init: Optional[float] = None,
) -> float:
    """
    Search the proper coupling strength g by plotting the dispersive shift of ground and excited state vs. flux
    """

    # Pre-calculate signal amplitude for plotting
    signal_amp = np.abs(signals)

    STEP = 0.001

    if g_init is None:
        g_init = round(0.5 * (g_bound[0] + g_bound[1]), 3)

    @lru_cache(maxsize=None)
    def get_dispersive(g: float) -> Tuple[np.ndarray, ...]:
        # only use 4 states to calculate the ground state dispersive shift for speed
        return calculate_dispersive_vs_flx(
            params, sp_flxs, r_f, g, progress=False, resonator_dim=4
        )

    rf_0, rf_1 = get_dispersive(g_init)

    # Create slider widget
    g_slider = widgets.FloatSlider(
        value=g_init,
        min=g_bound[0],
        max=g_bound[1],
        step=STEP,
        description="g (GHz):",
        continuous_update=False,
        style={"description_width": "initial"},
        readout_format=".3f",
    )

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.tight_layout(pad=4.0)

    # Initial plot with empty data
    ax.imshow(
        signal_amp.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=(sp_flxs[0], sp_flxs[-1], sp_fpts[0], sp_fpts[-1]),
        cmap="viridis",
    )

    (line_g,) = ax.plot(sp_flxs, rf_0, "b-", label="Ground state")
    (line_e,) = ax.plot(sp_flxs, rf_1, "r-", label="Excited state")
    ax.axhline(y=r_f, color="k", linestyle="--", label="Bare resonator")

    ax.set_ylim(sp_fpts.min(), sp_fpts.max())
    ax.set_xlabel(r"Flux $\Phi_{ext}/\Phi_0$")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title(f"g = {g_init:.3f} GHz")
    ax.legend(loc="upper right")

    # Register callback for slider changes
    def on_g_change(change):
        g = change["new"]

        rf_0, rf_1 = get_dispersive(g)
        # rf_list = get_dispersive(g)

        # Update the lines
        line_g.set_data(sp_flxs, rf_0)
        line_e.set_data(sp_flxs, rf_1)
        # for i, line in enumerate(line_list):
        #     line.set_data(sp_flxs, rf_list[i])

        # Update the title with current g value
        ax.set_title(f"g = {g:.3f} GHz")

    g_slider.observe(on_g_change, names="value")

    # Layout everything together
    display(g_slider)

    def close_and_get_g():
        plt.draw()
        plt.pause(0.5)
        plt.close(fig)
        return g_slider.value

    return close_and_get_g


def auto_fit_dispersive(
    params: Tuple[float, float, float],
    r_f: float,
    sp_flxs: np.ndarray,
    sp_fpts: np.ndarray,
    signals: np.ndarray,
    g_bound: Tuple[float, float] = (0.01, 0.2),
    g_init: Optional[float] = None,
    fit_rf: bool = False,
) -> float:
    """
    Auto fit the coupling strength g by maximizing the overlap of predicted ground state frequency with onetone spectrum
    """

    MAX_ITER = 1000

    pbar = tqdm(total=MAX_ITER, desc="Auto fitting g", leave=False)

    def update_pbar(g):
        pbar.update(1)
        pbar.set_postfix_str(f"g = {g:.3f} GHz")

    amps = np.abs(signals)

    # determine whether fit the value to max or min
    fit_factor = 1 if np.sum(amps[:, amps.shape[1] // 2] - amps[:, 0]) < 0 else -1

    # derive the initial g value if not provided
    if g_init is None:
        g_init = 0.5 * (g_bound[0] + g_bound[1])

    # derive the fitting tolerance
    ftol = np.max(amps) * 1e-4

    def loss_fn(g, r_f):
        update_pbar(g)

        # only use 4 states to calculate the ground state dispersive shift for speed
        rf_0, *_ = calculate_dispersive_vs_flx(
            params, sp_flxs, r_f, g, progress=False, resonator_dim=4
        )

        # 用線性插值取得每個 rf_0 對應的 signal
        vals = [np.interp(rf, sp_fpts, amps[i]) for i, rf in enumerate(rf_0)]
        return -fit_factor * np.mean(vals)

    fit_kwargs = dict(
        method="L-BFGS-B",
        options={"disp": False, "maxiter": MAX_ITER, "ftol": ftol},
    )
    if fit_rf:
        res = minimize(
            lambda p: loss_fn(p[0], p[1]),
            x0=[g_init, r_f],
            bounds=[g_bound, [r_f - 2e-3, r_f + 2e-3]],
            **fit_kwargs,
        )
        if not isinstance(res, np.ndarray):
            res = res.x  # compatibility with scipy < 1.7

        return res[0].item(), res[1].item()
    else:
        res = minimize(
            lambda p: loss_fn(p[0], r_f), x0=[g_init], bounds=[g_bound], **fit_kwargs
        )
        if not isinstance(res, np.ndarray):
            res = res.x  # compatibility with scipy < 1.7

        return res[0].item(), None


def plot_dispersive_with_onetone(
    r_f: float,
    g: float,
    mAs: np.ndarray,
    flxs: np.ndarray,
    rf_list: Tuple[np.ndarray, ...],
    sp_mAs: np.ndarray,
    sp_flxs: np.ndarray,
    sp_fpts: np.ndarray,
    signals: np.ndarray,
) -> go.Figure:
    """
    Plot the dispersive resonator frequency vs. flux with one tone signal
    Contain the ground and excited state dispersive shift
    """
    fig = go.Figure()

    # Add the signal as a heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.abs(signals).T,
            x=sp_mAs,
            y=sp_fpts,
            colorscale="Viridis",
            showscale=False,  # Disable the color bar
        )
    )

    # Add the qubit at 0 and 1 as line plots
    colors = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray"]
    if len(rf_list) <= len(colors):
        kwargs = [dict(line=dict(color=colors[i])) for i in range(len(rf_list))]
    else:
        kwargs = [dict()] * len(rf_list)  # too many states, use default color

    for i, rf in enumerate(rf_list):
        fig.add_trace(
            go.Scatter(x=mAs, y=rf, mode="lines", name=f"state {i}", **kwargs[i])
        )

    # plot a dash hline to indicate the 0 point, also add a xaxis2 to show mA
    fig.add_scatter(
        x=sp_mAs,
        y=np.full_like(sp_mAs, r_f),
        xaxis="x2",
        line=dict(color="black", dash="dash"),
        name="origin",
    )
    mAs_ticks = sp_mAs[:: max(1, len(sp_mAs) // 20)]
    flxs_ticks = sp_flxs[:: max(1, len(sp_flxs) // 20)]
    fig.update_layout(
        xaxis2=dict(
            tickvals=mAs_ticks,
            ticktext=[f"{flx:.2f}" for flx in flxs_ticks],
            matches="x1",
            overlaying="x1",
            side="top",
            title_text="$Φ_{ext}/Φ_0$",
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title="mA",
        yaxis_title="Frequency (GHz)",
        legend_title=f"g = {g:.3f} GHz",
        margin=dict(l=0, r=0, t=30, b=0),
    )

    fig.update_yaxes(range=[sp_fpts.min(), sp_fpts.max()])

    return fig
