"""Matplotlib diagnostic view for a fluxonium database search."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.search import DatabaseSearchResult


def make_search_diagnostic_figure(result: DatabaseSearchResult) -> Figure:
    """Create the pyplot-managed diagnostic Figure without showing it."""
    fig = plt.figure(figsize=(10, 7))
    assert isinstance(fig, Figure)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

    fig.suptitle(
        f"Best Distance: {result.best_distance:.2g}, EJ={result.params[0]:.2f}, EC={result.params[1]:.2f}, EL={result.params[2]:.2f}"
    )

    if result.predicted_freqs.size == 0:
        raise TypeError("'numpy.float64' object cannot be interpreted as an integer")

    # Frequency comparison plot
    ax_freq = fig.add_subplot(gs[:, 0])
    ax_freq.scatter(
        result.fluxs, result.freqs, label="Target", color="blue", marker="o"
    )
    ax_freq.scatter(
        result.fluxs, result.predicted_freqs, label="Predicted", color="red", marker="x"
    )
    ax_freq.set_ylabel("Frequency (GHz)")
    ax_freq.set_xlabel("Flux")
    ax_freq.legend()
    ax_freq.grid(True)

    # Per-parameter distance scatter (4k points each — rasterize).
    dists, scales = result.entry_results[:, 0], result.entry_results[:, 1]
    finite_dists = dists[np.isfinite(dists)]
    y_top = float(np.max(finite_dists) * 1.1) if finite_dists.size else None
    for i, (name, bound) in enumerate(
        [("EJ", result.bounds.EJ), ("EC", result.bounds.EC), ("EL", result.bounds.EL)]
    ):
        ax_param = fig.add_subplot(gs[i, 1])
        ax_param.set_xlim(*bound)
        ax_param.set_xlabel(name)
        ax_param.set_ylabel("Distance")
        ax_param.grid()

        ax_param.scatter(
            result.entry_params[:, i] * scales, dists, s=2, rasterized=True
        )
        ax_param.scatter(
            [result.params[i]], [result.best_distance], color="red", s=50, marker="*"
        )
        if y_top is not None:
            ax_param.set_ylim(0.0, y_top)

    return fig
