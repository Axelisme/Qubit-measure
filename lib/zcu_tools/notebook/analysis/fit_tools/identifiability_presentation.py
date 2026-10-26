"""Figure and table views of shared identifiability records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .identifiability import (
    PROFILE_DELTA_CHI2_THRESHOLD,
    FitCorrelation,
    ParameterProfile,
    WeakParameter,
)


def plot_fit_correlation(
    info: FitCorrelation,
    *,
    labels: Mapping[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """Return an annotated correlation heatmap, naming log-covariance space.

    labels optionally maps parameter names to physical-unit display labels.
    The caller owns display, saving and closing the returned figure.
    """
    names = [
        name if labels is None else labels.get(name, name)
        for name in info.parameter_names
    ]
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


def plot_parameter_profile(
    profile: ParameterProfile,
    *,
    label: str | None = None,
    scale: float = 1.0,
) -> tuple[Figure, Axes]:
    """Plot a profile, threshold, best value and finite physical fit bounds.

    label names the x-axis; scale converts physical values to display units
    and must be positive and finite. The caller owns the returned figure.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("display scale must be positive and finite")
    fig, ax = plt.subplots()
    ax.plot(profile.values * scale, profile.delta_chi2, marker=".", label="Profile")
    ax.axhline(PROFILE_DELTA_CHI2_THRESHOLD, ls="--", color="gray", label="Δχ² = 1")
    ax.axvline(profile.best_value * scale, color="black", label="Best fit")
    for bound, name in zip(profile.bounds, ("Lower bound", "Upper bound"), strict=True):
        if np.isfinite(bound):
            ax.axvline(bound * scale, ls=":", color="gray", label=name)
    ax.set_xscale("log")
    ax.set_xlabel(profile.parameter if label is None else label)
    ax.set_ylabel("Δχ²")
    ax.set_title("Profile (natural-log covariance scaling)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def make_weak_parameter_table(weak: Sequence[WeakParameter]) -> pd.DataFrame:
    """Return parameter/reasons columns; empty input returns an empty table."""
    return pd.DataFrame(
        [(entry.parameter, "; ".join(entry.reasons)) for entry in weak],
        columns=["parameter", "reasons"],
    )
