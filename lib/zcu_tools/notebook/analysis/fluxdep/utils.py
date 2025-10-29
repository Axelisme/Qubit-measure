from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from zcu_tools.simulate import mA2flx

from .models import energy2transition
from .processing import cast2real_and_norm


def add_secondary_xaxis(
    fig: go.Figure, flxs: np.ndarray, mAs: np.ndarray, num_ticks: int = 12
):
    # Choose a small number of evenly spaced tick positions by index
    n = flxs.shape[0]
    if n <= num_ticks:
        tick_indices = np.arange(n)
    else:
        tick_indices = np.linspace(0, n - 1, num_ticks)
        tick_indices = np.round(tick_indices).astype(int)
        tick_indices = np.unique(tick_indices)

    xticks = flxs[tick_indices]
    mA_ticks = mAs[tick_indices]

    # a invisible trace to make the xaxis2 show
    fig.add_trace(
        go.Scatter(
            x=xticks,
            y=np.zeros_like(xticks),
            mode="lines",
            opacity=0.0,
            xaxis="x2",
            showlegend=False,
            zorder=0,
        )
    )
    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=xticks.tolist(),
            ticktext=[f"{val:.1e}" for val in mA_ticks],
            showgrid=False,
        ),
    )


class FluxDependVisualizer:
    def __init__(self, fig: Optional[go.Figure] = None) -> None:
        if fig is None:
            fig = go.Figure()
        self.fig = fig

    def get_figure(self) -> go.Figure:
        return self.fig

    def add_secondary_xaxis(
        self, flxs: np.ndarray, mAs: np.ndarray
    ) -> "FluxDependVisualizer":
        add_secondary_xaxis(self.fig, flxs, mAs)
        return self


class FreqFluxDependVisualizer(FluxDependVisualizer):
    def __init__(self, fig: Optional[go.Figure] = None) -> None:
        super().__init__(fig)

        self.xlimits = [np.inf, -np.inf]
        self.ylimits = [np.inf, -np.inf]

    def update_limits(
        self,
        xrange: Tuple[Union[float, None], Union[float, None]],
        yrange: Tuple[Union[float, None], Union[float, None]],
    ) -> None:
        self.xlimits = [
            self.xlimits[0] if xrange[0] is None else min(self.xlimits[0], xrange[0]),
            self.xlimits[1] if xrange[1] is None else max(self.xlimits[1], xrange[1]),
        ]
        self.ylimits = [
            self.ylimits[0] if yrange[0] is None else min(self.ylimits[0], yrange[0]),
            self.ylimits[1] if yrange[1] is None else max(self.ylimits[1], yrange[1]),
        ]

    def plot_background(
        self, spects: Dict[str, Dict[str, Any]]
    ) -> "FreqFluxDependVisualizer":
        # Add heatmap traces for each spectrum in spects
        for name, spect in spects.items():
            # Get corresponding data and range
            signals = spect["spectrum"]["data"]
            flx_mask = np.any(~np.isnan(signals), axis=1)
            fpt_mask = np.any(~np.isnan(signals), axis=0)
            signals = signals[flx_mask, :][:, fpt_mask]
            values = spect["spectrum"]["mAs"][flx_mask]
            fpts = spect["spectrum"]["fpts"][fpt_mask]

            # Normalize data
            norm_signals = cast2real_and_norm(signals)

            # convert values to flxs
            flxs = mA2flx(values, spect["mA_c"], spect["period"])

            # Add heatmap trace
            self.fig.add_trace(
                go.Heatmap(
                    z=norm_signals.T,
                    x=flxs,
                    y=fpts,
                    colorscale="Greys",
                    showscale=False,
                    meta=name,
                    zorder=0,
                )
            )

            # update xlimits/ylimits
            self.update_limits((flxs.min(), flxs.max()), (fpts.min(), fpts.max()))

        return self

    def plot_simulation_lines(
        self, flxs: np.ndarray, energies: np.ndarray, allows: Dict[str, Any]
    ) -> "FreqFluxDependVisualizer":
        fs, labels = energy2transition(energies, allows)

        for i, label in enumerate(labels):
            self.fig.add_trace(
                go.Scatter(x=flxs, y=fs[:, i], mode="lines", name=label, zorder=1)
            )

        return self

    def plot_points(
        self, flxs: np.ndarray, ys: np.ndarray, **kwargs
    ) -> "FreqFluxDependVisualizer":
        self.fig.add_trace(
            go.Scatter(
                x=flxs,
                y=ys,
                mode="markers",
                zorder=2,
                showlegend=False,
                **kwargs,
            )
        )

        self.update_limits((flxs.min(), flxs.max()), (ys.min(), ys.max()))

        return self

    def plot_sample_points(
        self, sample_table: pd.DataFrame, convert_fn: Callable[[float], float]
    ) -> "FreqFluxDependVisualizer":
        xs = convert_fn(sample_table["calibrated mA"])
        ys = sample_table["Freq (MHz)"] * 1e-3  # GHz
        labels = [
            ", ".join(
                [
                    f"{key}={value}"
                    for key, value in row.items()
                    if key not in ["calibrated mA", "Freq (MHz)"]
                ]
            )
            for _, row in sample_table.iterrows()
        ]
        self.plot_points(xs, ys, opacity=0.5, hovertext=labels)

        self.update_limits((xs.min(), xs.max()), (ys.min(), ys.max()))

        return self

    def add_constant_freq(self, freq: float, name: str) -> "FreqFluxDependVisualizer":
        self.fig.add_hline(y=freq, line_dash="dash", name=name)

        self.update_limits((None, None), (freq - 1e-6, freq + 1e-6))

        return self

    def auto_derive_limits(self) -> "FreqFluxDependVisualizer":
        self.fig.update_layout(
            xaxis=dict(range=self.xlimits),
            yaxis=dict(range=self.ylimits),
        )

        return self
