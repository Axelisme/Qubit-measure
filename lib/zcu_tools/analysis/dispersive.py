import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scqubits as scq
from IPython.display import display
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .general import figsize


def dispersive1D_analyze(xs, snrs, xlabel=None):
    """
    Analyze 1D dispersive measurement data to find the maximum SNR position.

    This function processes 1D signal-to-noise ratio data, applies Gaussian smoothing,
    finds the position of maximum SNR, and visualizes the results.

    Parameters
    ----------
    xs : array-like
        X-axis values (e.g., frequency, power, etc.)
    snrs : array-like
        Signal-to-noise ratio values corresponding to xs
    xlabel : str, optional
        Label for the x-axis in the plot

    Returns
    -------
    float
        The x-value corresponding to the maximum SNR
    """
    snrs = np.abs(snrs)

    # fill NaNs with zeros
    snrs[np.isnan(snrs)] = 0

    snrs = gaussian_filter1d(snrs, 1)

    max_id = np.argmax(snrs)
    max_x = xs[max_id]

    plt.figure(figsize=figsize)
    plt.plot(xs, snrs)
    plt.axvline(max_x, color="r", ls="--", label=f"max SNR = {snrs[max_id]:.2f}")
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.ylabel("SNR (a.u.)")
    plt.legend()
    plt.show()

    return max_x


def dispersive2D_analyze(xs, ys, snr2D, xlabel=None, ylabel=None):
    """
    Analyze 2D dispersive measurement data to find the maximum SNR position.

    This function processes 2D signal-to-noise ratio data, applies Gaussian smoothing,
    finds the position of maximum SNR in the 2D space, and visualizes the results as a heatmap.

    Parameters
    ----------
    xs : array-like
        X-axis values (e.g., frequency, power, etc.)
    ys : array-like
        Y-axis values (e.g., time, bias, etc.)
    snr2D : 2D array-like
        2D grid of signal-to-noise ratio values
    xlabel : str, optional
        Label for the x-axis in the plot
    ylabel : str, optional
        Label for the y-axis in the plot

    Returns
    -------
    tuple
        (x_max, y_max) coordinates corresponding to the maximum SNR
    """
    snr2D = np.abs(snr2D)

    # fill NaNs with zeros
    snr2D[np.isnan(snr2D)] = 0

    snr2D = gaussian_filter(snr2D, 1)

    x_max_id = np.argmax(np.max(snr2D, axis=0))
    y_max_id = np.argmax(np.max(snr2D, axis=1))
    x_max = xs[x_max_id]
    y_max = ys[y_max_id]

    plt.figure(figsize=figsize)
    plt.imshow(
        snr2D,
        aspect="auto",
        origin="lower",
        interpolation="none",
        extent=(xs[0], xs[-1], ys[0], ys[-1]),
    )
    plt.scatter(
        x_max, y_max, color="r", label=f"max SNR = {snr2D[y_max_id, x_max_id]:.2e}"
    )
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.colorbar(label="SNR (a.u.)")
    plt.legend()
    plt.show()

    return x_max, y_max


def ge_lookback_analyze(Ts, signals_g, signals_e, *, pulse_cfg=None, smooth=None):
    """
    Analyze ground and excited state signals over time to evaluate measurement contrast.

    This function processes time-series data for ground and excited state signals,
    applies Gaussian smoothing, calculates the magnitude and contrast between states,
    and visualizes the results.

    Parameters
    ----------
    Ts : array-like
        Time points in microseconds
    signal_g : array-like
        Signal data for ground state
    signal_e : array-like
        Signal data for excited state
    pulse_cfg : dict, optional
        Configuration parameters for the pulse (not used in current implementation)

    Returns
    -------
    None
        This function only produces a plot but doesn't return a value
    """
    if smooth is not None:
        signals_g = gaussian_filter1d(signals_g, smooth)
        signals_e = gaussian_filter1d(signals_e, smooth)

    amps_g = np.abs(signals_g)
    amps_e = np.abs(signals_e)
    contrast = np.abs(signals_g - signals_e)

    plt.figure(figsize=figsize)
    plt.plot(Ts, amps_g, label="g")
    plt.plot(Ts, amps_e, label="e")
    plt.plot(Ts, contrast, label="contrast")

    if pulse_cfg is not None:
        trig_offset = pulse_cfg["trig_offset"]
        ro_length = pulse_cfg["ro_length"]
        plt.axvline(trig_offset, color="g", linestyle="--", label="ro start")
        plt.axvline(trig_offset + ro_length, color="g", linestyle="--", label="ro end")

    plt.xlabel("Time (us)")
    plt.ylabel("Magnitude (a.u.)")
    plt.legend()
    plt.show()


def calculate_dispersive(
    params, r_f, g, flxs, cutoff=50, evals_count=40, progress=True
):
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


def search_proper_g(params, r_f, sp_flxs, sp_fpts, signals, g_bound):
    # Pre-calculate signal amplitude for plotting
    signal_amp = np.abs(signals)

    STEP = 0.001
    cache_rfs = {}

    g_init = round(0.5 * (g_bound[0] + g_bound[1]), 3)
    rf_0, rf_1 = calculate_dispersive(
        params, r_f, g_init, sp_flxs, cutoff=40, evals_count=20, progress=False
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
    ax.set_title(f"g = {g_init:.4f} GHz")
    ax.legend(loc="upper right")

    # Register callback for slider changes
    def on_g_change(change):
        g = change["new"]

        if g in cache_rfs:
            rf_0, rf_1 = cache_rfs[g]
        else:
            # Calculate dispersive shift
            rf_0, rf_1 = calculate_dispersive(
                params, r_f, g, sp_flxs, cutoff=40, evals_count=20, progress=False
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
        plt.close(fig)
        return g_slider.value

    return close_and_get_g


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
