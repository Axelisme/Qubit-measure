import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from scipy.optimize import minimize
from tqdm.auto import tqdm


def calculate_dispersive(
    params, r_f, g, flxs, cutoff=50, evals_count=40, progress=True
):
    import scqubits as scq

    resonator = scq.Oscillator(r_f, truncated_dim=2, id_str="resonator")
    fluxonium = scq.Fluxonium(
        *params,
        flux=0.5,
        cutoff=cutoff,
        truncated_dim=int(evals_count / 2 + 0.6),
        id_str="qubit",
    )
    hilbertspace = scq.HilbertSpace([resonator, fluxonium])
    hilbertspace.add_interaction(
        g=g,
        op1=resonator.creation_operator,
        op2=fluxonium.n_operator,
        add_hc=True,
        id_str="q-r coupling",
    )

    def update_hilbertspace(flux):
        fluxonium.flux = flux

    old, scq.settings.PROGRESSBAR_DISABLED = (
        scq.settings.PROGRESSBAR_DISABLED,
        not progress,
    )
    sweep = scq.ParameterSweep(
        hilbertspace,
        {"flxs": flxs},
        update_hilbertspace=update_hilbertspace,
        evals_count=evals_count,
        subsys_update_info={"flxs": [fluxonium]},
        labeling_scheme="LX",
    )
    scq.settings.PROGRESSBAR_DISABLED = old

    evals = sweep["evals"].toarray()

    flx_idxs = np.arange(len(flxs))
    idx_00 = sweep.dressed_index((0, 0)).toarray()
    idx_10 = sweep.dressed_index((1, 0)).toarray()
    idx_01 = sweep.dressed_index((0, 1)).toarray()
    idx_11 = sweep.dressed_index((1, 1)).toarray()

    rf_0 = evals[flx_idxs, idx_10] - evals[flx_idxs, idx_00]
    rf_1 = evals[flx_idxs, idx_11] - evals[flx_idxs, idx_01]

    return rf_0, rf_1


def search_proper_g(params, r_f, sp_flxs, sp_fpts, signals, g_bound, g_init=None):
    # Pre-calculate signal amplitude for plotting
    signal_amp = np.abs(signals)

    CUTOFF = 40
    EVALS_COUNT = 20
    STEP = 0.001
    cache_rfs = {}

    if g_init is None:
        g_init = round(0.5 * (g_bound[0] + g_bound[1]), 3)
    rf_0, rf_1 = calculate_dispersive(
        params,
        r_f,
        g_init,
        sp_flxs,
        cutoff=CUTOFF,
        evals_count=EVALS_COUNT,
        progress=False,
    )
    cache_rfs[g_init] = (rf_0, rf_1)

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
    fig.tight_layout()

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

        if g in cache_rfs:
            rf_0, rf_1 = cache_rfs[g]
        else:
            # Calculate dispersive shift
            rf_0, rf_1 = calculate_dispersive(
                params,
                r_f,
                g,
                sp_flxs,
                cutoff=CUTOFF,
                evals_count=EVALS_COUNT,
                progress=False,
            )
            cache_rfs[g] = (rf_0, rf_1)

        # Update the lines
        line_g.set_data(sp_flxs, rf_0)
        line_e.set_data(sp_flxs, rf_1)

        # Update the title with current g value
        ax.set_title(f"g = {g:.3f} GHz")

    g_slider.observe(on_g_change, names="value")

    # Layout everything together
    display(g_slider)

    def close_and_get_g():
        plt.draw()
        plt.pause(0.1)
        plt.close(fig)
        return g_slider.value

    return close_and_get_g


def auto_fit_dispersive(
    params, r_f, sp_flxs, sp_fpts, signals, g_bound=(0.01, 0.2), g_init=None
):
    MAX_ITER = 1000

    pbar = tqdm(total=MAX_ITER, desc="Auto fitting g", leave=False)

    def update_pbar(g):
        pbar.update(1)
        pbar.set_postfix_str(f"g = {g:.3f} GHz")

    amps = np.abs(signals)

    def loss(g):
        # g 是 array, 取第一個元素
        g_val = float(g[0])

        update_pbar(g_val)

        rf_0, _ = calculate_dispersive(
            params, r_f, g_val, sp_flxs, cutoff=40, evals_count=20, progress=False
        )

        # 用線性插值取得每個 rf_0 對應的 signal
        vals = [np.interp(rf, sp_fpts, amps[i]) for i, rf in enumerate(rf_0)]
        return -np.mean(vals)

    ftol = np.max(amps) * 1e-4

    if g_init is None:
        g_init = 0.5 * (g_bound[0] + g_bound[1])
    res = minimize(
        loss,
        x0=[g_init],
        bounds=[g_bound],
        method="L-BFGS-B",
        options={"disp": False, "maxiter": MAX_ITER, "ftol": ftol},
    )
    if not isinstance(res, np.ndarray):
        res = res.x  # compatibility with scipy < 1.7

    return res[0].item()


def plot_dispersive_with_onetone(
    r_f, g, mAs, flxs, rf_0, rf_1, sp_mAs, sp_flxs, sp_fpts, signals
):
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
    fig.add_trace(
        go.Scatter(
            x=mAs,
            y=rf_0,
            mode="lines",
            line=dict(color="blue"),
            name="ground",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mAs,
            y=rf_1,
            mode="lines",
            line=dict(color="red"),
            name="excited",
        )
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
