from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray
from typing_extensions import Callable, Optional, Self, Union

from zcu_tools.notebook.persistance import SpectrumResult, TransitionDict

from .models import energy2transition
from .processing import cast2real_and_norm


def derive_bound(
    spectrums: dict[str, SpectrumResult],
    convert_fn: Callable[[SpectrumResult], NDArray[np.float64]],
):
    return (
        np.nanmin([np.nanmin(convert_fn(s)) for s in spectrums.values()]),
        np.nanmax([np.nanmax(convert_fn(s)) for s in spectrums.values()]),
    )


def add_secondary_xaxis(
    fig: go.Figure,
    first_xs: NDArray[np.float64],
    second_xs: NDArray[np.float64],
    num_ticks: int = 12,
    **fig_kwargs,
) -> None:
    # Choose a small number of evenly spaced tick positions by index
    n = first_xs.shape[0]
    if n <= num_ticks:
        tick_indices = np.arange(n)
    else:
        tick_indices = np.linspace(0, n - 1, num_ticks)
        tick_indices = np.round(tick_indices).astype(int)
        tick_indices = np.unique(tick_indices)

    first_ticks = first_xs[tick_indices]
    second_ticks = second_xs[tick_indices]

    # a invisible trace to make the xaxis2 show
    fig.add_trace(
        go.Scatter(
            x=first_ticks,
            y=np.zeros_like(first_ticks),
            mode="lines",
            opacity=0.0,
            xaxis="x2",
            showlegend=False,
            zorder=0,
        ),
        **fig_kwargs,
    )
    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=first_ticks.tolist(),
            ticktext=[f"{val:.1e}" for val in second_ticks],
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

    def add_dev_values_ticks(
        self, fluxs: NDArray[np.float64], dev_values: NDArray[np.float64]
    ) -> Self:
        add_secondary_xaxis(self.fig, fluxs, dev_values)
        return self


class FreqFluxDependVisualizer(FluxDependVisualizer):
    def __init__(self, fig: Optional[go.Figure] = None) -> None:
        super().__init__(fig)

        self.xlimits = [np.inf, -np.inf]
        self.ylimits = [np.inf, -np.inf]

    def update_limits(
        self,
        xrange: tuple[Union[float, None], Union[float, None]],
        yrange: tuple[Union[float, None], Union[float, None]],
    ) -> None:
        self.xlimits = [
            self.xlimits[0] if xrange[0] is None else min(self.xlimits[0], xrange[0]),
            self.xlimits[1] if xrange[1] is None else max(self.xlimits[1], xrange[1]),
        ]
        self.ylimits = [
            self.ylimits[0] if yrange[0] is None else min(self.ylimits[0], yrange[0]),
            self.ylimits[1] if yrange[1] is None else max(self.ylimits[1], yrange[1]),
        ]

    def plot_background(self, spectrums: dict[str, SpectrumResult]) -> Self:
        # Add heatmap traces for each spectrum in spects
        for name, spectrum in spectrums.items():
            # Get corresponding data and range
            spect_data = spectrum["spectrum"]
            signals = spect_data["signals"]
            flux_mask = np.any(~np.isnan(signals), axis=1)
            freq_mask = np.any(~np.isnan(signals), axis=0)
            signals = signals[flux_mask, :][:, freq_mask]
            fluxs: NDArray[np.float64] = spect_data["fluxs"][flux_mask]
            freqs: NDArray[np.float64] = spect_data["freqs"][freq_mask]

            # Normalize data
            real_signals = cast2real_and_norm(signals)

            # Add heatmap trace
            self.fig.add_trace(
                go.Heatmap(
                    z=real_signals.T,
                    x=fluxs,
                    y=freqs,
                    colorscale="Greys",
                    showscale=False,
                    meta=name,
                    zorder=0,
                )
            )

            # update xlimits/ylimits
            self.update_limits((fluxs.min(), fluxs.max()), (freqs.min(), freqs.max()))

        return self

    def plot_simulation_lines(
        self,
        fluxs: NDArray[np.float64],
        energies: NDArray[np.float64],
        transitions: TransitionDict,
    ) -> Self:
        freqs, labels = energy2transition(energies, transitions)

        for i, label in enumerate(labels):
            self.fig.add_trace(
                go.Scatter(x=fluxs, y=freqs[:, i], mode="lines", name=label, zorder=1)
            )

        return self

    def plot_points(
        self, fluxs: NDArray[np.float64], freqs: NDArray[np.float64], **kwargs
    ) -> Self:
        self.fig.add_trace(
            go.Scatter(
                x=fluxs,
                y=freqs,
                mode="markers",
                zorder=2,
                showlegend=False,
                **kwargs,
            )
        )

        self.update_limits((fluxs.min(), fluxs.max()), (freqs.min(), freqs.max()))

        return self

    def plot_sample_points(
        self,
        sample_table: pd.DataFrame,
        convert_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> Self:
        xs = convert_fn(np.asarray(sample_table["calibrated mA"]))
        ys = np.asarray(sample_table["Freq (MHz)"]) * 1e-3  # GHz
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

    def plot_constant_freq(self, freq: float, name: str) -> Self:
        self.fig.add_hline(y=freq, line_dash="dash", name=name)

        self.update_limits((None, None), (freq - 1e-6, freq + 1e-6))

        return self

    def set_limits(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> Self:
        self.fig.update_layout(xaxis=dict(range=xlim), yaxis=dict(range=ylim))
        return self

    def auto_derive_limits(self) -> Self:
        self.fig.update_layout(
            xaxis=dict(range=self.xlimits), yaxis=dict(range=self.ylimits)
        )

        return self
