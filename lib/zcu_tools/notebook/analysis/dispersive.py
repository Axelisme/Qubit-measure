from __future__ import annotations

from functools import lru_cache

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output, display
from numpy.typing import NDArray
from scipy.optimize import minimize
from tqdm.auto import tqdm
from typing_extensions import Callable, Optional

from zcu_tools.simulate.fluxonium import calculate_dispersive_vs_flux


def search_proper_g(
    params: tuple[float, float, float],
    bare_rf: float,
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    sp_signals: NDArray,
    g_bound: tuple[float, float],
    g_init: Optional[float] = None,
) -> Callable[[], tuple[float, float]]:
    """
    Search the proper coupling strength g and resonator frequency r_f by plotting the dispersive shift
    of ground and excited state vs. flux. Returns a function that when called, returns (g, r_f) values.
    """

    # Pre-calculate signal amplitude for plotting
    real_signals = np.abs(sp_signals)

    # Default parameter values
    default_qub_dim = 15
    default_qub_cutoff = 30
    default_res_dim = 4
    default_step = max(round(len(sp_fluxs) / 100), 1)

    if g_init is None:
        g_init = round(0.5 * (g_bound[0] + g_bound[1]), 1)

    # Create parameter input widgets
    qub_dim_input = widgets.IntText(
        value=default_qub_dim,
        description="qub_dim:",
        style={"description_width": "initial"},
    )

    qub_cutoff_input = widgets.IntText(
        value=default_qub_cutoff,
        description="qub_cutoff:",
        style={"description_width": "initial"},
    )

    res_dim_input = widgets.IntText(
        value=default_res_dim,
        description="res_dim:",
        style={"description_width": "initial"},
    )

    step_input = widgets.IntText(
        value=default_step,
        description="step:",
        style={"description_width": "initial"},
    )

    # finish button
    def on_finish(_):
        plt.close(fig)
        clear_output(wait=False)

    finish_button = widgets.Button(
        description="Finish", style={"description_width": "initial"}
    )
    finish_button.on_click(on_finish)

    # Create slider widgets
    g_MHz_slider = widgets.FloatSlider(
        value=1e3 * g_init,
        min=1e3 * g_bound[0],
        max=1e3 * g_bound[1],
        step=1.0,
        description="g (MHz):",
        continuous_update=False,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    # Calculate r_f slider parameters

    rf_MHz_slider = widgets.FloatSlider(
        value=bare_rf,
        min=1e3 * sp_freqs.min(),
        max=1e3 * sp_freqs.max(),
        step=1e3 * 0.5 * (sp_freqs.max() - sp_freqs.min()) / len(sp_freqs),
        description="r_f (MHz):",
        continuous_update=False,
        style={"description_width": "initial"},
        readout_format=".1f",
    )

    @lru_cache(maxsize=None)
    def get_dispersive(
        g: float,
        r_f: float,
        qub_dim: int = default_qub_dim,
        qub_cutoff: int = default_qub_cutoff,
        res_dim: int = default_res_dim,
        step: int = default_step,
    ) -> tuple[np.ndarray, ...]:
        # Calculate the dispersive shift using provided parameters
        return calculate_dispersive_vs_flux(
            params,
            sp_fluxs[::step],
            r_f,
            g,
            progress=False,
            res_dim=res_dim,
            qub_cutoff=qub_cutoff,
            qub_dim=qub_dim,
        )

    flux_step = step_input.value
    rf_0, rf_1 = get_dispersive(g_init, bare_rf)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.tight_layout(pad=4.0)

    # Initial plot with empty data
    ax.imshow(
        real_signals.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=(sp_fluxs[0], sp_fluxs[-1], 1e3 * sp_freqs[0], 1e3 * sp_freqs[-1]),
        cmap="viridis",
    )

    (line_g,) = ax.plot(sp_fluxs[::flux_step], 1e3 * rf_0, "b-", label="Ground state")
    (line_e,) = ax.plot(sp_fluxs[::flux_step], 1e3 * rf_1, "r-", label="Excited state")
    line_bare = ax.axhline(y=bare_rf, color="k", linestyle="--", label="Bare resonator")

    ax.set_ylim(1e3 * sp_freqs.min(), 1e3 * sp_freqs.max())
    ax.set_xlabel(r"Flux $\Phi_{ext}/\Phi_0$")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(f"g = {1e3 * g_init:.1f} MHz, r_f = {bare_rf:.1f} MHz")
    ax.legend(loc="upper right")

    # Register callback for slider changes
    def update_plot() -> None:
        cur_g = 1e-3 * g_MHz_slider.value
        cur_rf = 1e-3 * rf_MHz_slider.value

        flux_step = step_input.value
        rf_0, rf_1 = get_dispersive(
            cur_g,
            cur_rf,
            qub_dim_input.value,
            qub_cutoff_input.value,
            res_dim_input.value,
            flux_step,
        )

        # Update the lines
        line_g.set_data(sp_fluxs[::flux_step], 1e3 * rf_0)
        line_e.set_data(sp_fluxs[::flux_step], 1e3 * rf_1)
        line_bare.set_ydata([1e3 * cur_rf])

        # Update the title with current values
        ax.set_title(f"g = {1e3 * cur_g:.1f} MHz, r_f = {1e3 * cur_rf:.1f} MHz")

    # Add observers for all widgets
    g_MHz_slider.observe(lambda _: update_plot(), names="value")
    rf_MHz_slider.observe(lambda _: update_plot(), names="value")
    qub_dim_input.observe(lambda _: update_plot(), names="value")
    qub_cutoff_input.observe(lambda _: update_plot(), names="value")
    res_dim_input.observe(lambda _: update_plot(), names="value")
    step_input.observe(lambda _: update_plot(), names="value")

    # Layout everything together
    parameter_box = widgets.HBox(
        [qub_dim_input, qub_cutoff_input, res_dim_input, step_input]
    )
    display(
        widgets.VBox(
            [parameter_box, widgets.HBox([g_MHz_slider, rf_MHz_slider, finish_button])]
        )
    )

    def close_and_get_values() -> tuple[float, float]:
        on_finish(None)  # Close the plot and clear output
        return 1e-3 * g_MHz_slider.value, 1e-3 * rf_MHz_slider.value

    return close_and_get_values


def auto_fit_dispersive(
    params: tuple[float, float, float],
    bare_rf_GHz: float,
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    sp_signals: NDArray,
    g_bound: tuple[float, float] = (0.01, 0.2),
    g_init: Optional[float] = None,
    fit_bare_rf: bool = False,
) -> tuple[float, Optional[float]]:
    """
    Auto fit the coupling strength g by maximizing the overlap of predicted ground state frequency with onetone spectrum
    """

    MAX_ITER = 1000
    MAX_POINTS = 300

    if len(sp_fluxs) > MAX_POINTS:  # downsample the data
        indices = np.round(np.linspace(0, len(sp_fluxs) - 1, MAX_POINTS)).astype(int)
        sp_fluxs = sp_fluxs[indices]
        sp_signals = sp_signals[indices]

    pbar = tqdm(total=MAX_ITER, desc="Auto fitting g", leave=False)

    def update_pbar(g, bare_rf_GHz):
        pbar.update(1)
        postfix = f"g = {1e3 * g:.1f} MHz"
        if fit_bare_rf:
            postfix += f", r_f = {1e3 * bare_rf_GHz:.1f} MHz"
        pbar.set_postfix_str(postfix)

    real_signals = np.abs(sp_signals)

    # derive the initial g value if not provided
    if g_init is None:
        g_init = 0.5 * (g_bound[0] + g_bound[1])

    # derive the fitting tolerance
    ftol = np.max(real_signals) * 1e-4

    def loss_fn(g, bare_rf_GHz):
        update_pbar(g, bare_rf_GHz)

        # only use 4 states to calculate the ground state dispersive shift for speed
        rf_0, rf_1 = calculate_dispersive_vs_flux(
            params, sp_fluxs, bare_rf_GHz, g, progress=False, res_dim=4
        )

        # 用線性插值取得每個 rf_0 對應的 signal
        vals = [
            max(
                np.interp(rf_0, sp_freqs, real_signal),
                np.interp(rf_1, sp_freqs, real_signal),
            )
            for rf_0, rf_1, real_signal in zip(rf_0, rf_1, real_signals)
        ]
        return -np.mean(vals)

    fit_kwargs = dict(
        method="L-BFGS-B",
        options={"disp": False, "maxiter": MAX_ITER, "ftol": ftol},
    )
    if fit_bare_rf:
        res = minimize(
            lambda p: loss_fn(p[0], p[1]),
            x0=[g_init, bare_rf_GHz],
            bounds=[g_bound, [bare_rf_GHz - 2e-3, bare_rf_GHz + 2e-3]],
            **fit_kwargs,
        )
        if not isinstance(res, np.ndarray):
            res = res.x  # compatibility with scipy < 1.7

        return res[0].item(), res[1].item()
    else:
        res = minimize(
            lambda p: loss_fn(p[0], bare_rf_GHz),
            x0=[g_init],
            bounds=[g_bound],
            **fit_kwargs,
        )
        if not isinstance(res, np.ndarray):
            res = res.x  # compatibility with scipy < 1.7

        return res[0].item(), None


def plot_dispersive_with_onetone(
    bare_rf: float,
    g: float,
    fluxs: NDArray[np.float64],
    plot_rfs: tuple[NDArray[np.float64], ...],
    sp_fluxs: NDArray[np.float64],
    sp_freqs: NDArray[np.float64],
    sp_signals: NDArray,
) -> go.Figure:
    """
    Plot the dispersive resonator frequency vs. flux with one tone signal
    Contain the ground and excited state dispersive shift
    """
    fig = go.Figure()

    # Add the signal as a heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.abs(sp_signals).T,
            x=sp_fluxs,
            y=sp_freqs,
            colorscale="Viridis",
            showscale=False,  # Disable the color bar
        )
    )

    # Add the qubit at 0 and 1 as line plots
    colors = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray"]
    if len(plot_rfs) <= len(colors):
        kwargs = [dict(line=dict(color=colors[i])) for i in range(len(plot_rfs))]
    else:
        kwargs = [dict()] * len(plot_rfs)  # too many states, use default color

    for i, rf in enumerate(plot_rfs):
        rf = rf.copy()
        # rf[np.abs(np.diff(rf, prepend=rf[0])) > 0.5 * np.ptp(sp_freqs)] = np.nan
        diff_rf = np.diff(rf, prepend=rf[0])
        for j in range(1, len(diff_rf) - 1):
            sign_j = np.sign(diff_rf[j])
            sign_prev = np.sign(diff_rf[j - 1])
            sign_next = np.sign(diff_rf[j + 1])
            if (
                sign_j != sign_prev
                and sign_j != sign_next
                and np.abs(diff_rf[j]) > 0.01 * np.ptp(sp_freqs)
            ):
                rf[j] = np.nan  # remove points with large jump
        fig.add_trace(
            go.Scatter(x=fluxs, y=rf, mode="lines", name=f"state {i}", **kwargs[i])
        )

    # plot a dash hline to indicate the 0 point, also add a xaxis2 to show mA
    fig.add_scatter(
        x=fluxs,
        y=np.full_like(fluxs, bare_rf),
        line=dict(color="black", dash="dash"),
        name="origin",
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Flux (Φ/Φ₀)",
        yaxis_title="Frequency (GHz)",
        legend_title=f"g = {g:.3f} GHz",
        margin=dict(l=0, r=0, t=30, b=0),
    )

    fig.update_yaxes(range=[sp_freqs.min(), sp_freqs.max()])

    return fig
