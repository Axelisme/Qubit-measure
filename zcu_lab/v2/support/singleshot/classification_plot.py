"""Draw the GE Post classification overlay on a caller-owned IQ axes."""

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_ge_classification_boundary(
    ax: Axes, g_center: complex, e_center: complex, radius: float
) -> None:
    """Overlay the GE radius classifier on an autoscaled raw IQ axes.

    Supply the distinct finite g/e centers and nonnegative finite radius used
    for classification, in the same units as the IQ image. Draw both centers
    and the radius boundaries; overlapping disks share one bisector segment.
    Extend the auto limits to include the boundaries with a margin. This only
    draws the geometry; it does not classify the IQ data.
    Invalid geometry raises ValueError.
    """
    if not np.isfinite([g_center, e_center, radius]).all() or radius < 0:
        raise ValueError(
            "GE classification geometry requires finite centers and a nonnegative radius"
        )
    separation = abs(e_center - g_center)
    if separation == 0:
        raise ValueError("GE classification centers must be distinct")
    direction = (e_center - g_center) / separation
    cut = np.arccos(separation / (2 * radius)) if radius > separation / 2 else 0.0
    angles = np.linspace(cut, 2 * np.pi - cut, 361)
    directions = np.exp(1j * angles) * direction
    ground = g_center + radius * directions
    excited = e_center - radius * directions
    _plot_region_edge(ax, ground)
    _plot_region_edge(ax, excited)
    if cut > 0:
        # The ends of the ground arc are the two circle intersections.
        _plot_region_edge(ax, ground[[0, -1]])

    # Keep the centers distinguishable when the fitted radius spans only a few pixels.
    ax.plot(
        g_center.real,
        g_center.imag,
        "o",
        color="0.15",
        markerfacecolor="white",
        markersize=2.5,
        markeredgewidth=0.7,
    )
    ax.plot(
        e_center.real,
        e_center.imag,
        "x",
        color="0.15",
        markersize=3,
        markeredgewidth=0.8,
        path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")],
    )
    # An IQ image has sticky edges; do not let them obscure a coincident arc.
    ax.use_sticky_edges = False
    ax.autoscale_view()


def _plot_region_edge(ax: Axes, samples: NDArray[np.complex128]) -> None:
    ax.plot(
        samples.real,
        samples.imag,
        "--",
        color="0.15",
        linewidth=0.9,
        path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")],
    )
