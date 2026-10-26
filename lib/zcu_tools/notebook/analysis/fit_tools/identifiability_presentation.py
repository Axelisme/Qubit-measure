"""Figure and table views of shared identifiability records."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .identifiability import FitCorrelation, WeakParameter


def plot_fit_correlation(
    info: FitCorrelation,
) -> tuple[Figure, Axes]:
    """Return an annotated correlation heatmap, naming log-covariance space.

    Row/column labels follow info.parameter_names. The caller owns display,
    saving and closing the returned figure.
    """
    names = info.parameter_names
    fig, ax = plt.subplots()
    image = ax.imshow(info.correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    for row in range(len(names)):
        for col in range(len(names)):
            ax.text(
                col, row, f"{info.correlation[row, col]:.2f}", ha="center", va="center"
            )
    ax.set_title("Correlation (covariance in natural-log parameter space)")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig, ax


def make_weak_parameter_table(weak: Sequence[WeakParameter]) -> pd.DataFrame:
    """Return parameter/reasons columns; empty input returns an empty table."""
    return pd.DataFrame(
        [(entry.parameter, "; ".join(entry.reasons)) for entry in weak],
        columns=["parameter", "reasons"],
    )
