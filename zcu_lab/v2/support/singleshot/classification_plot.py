"""Draw GE Post classified density and boundaries on caller-owned IQ axes."""

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from numpy.typing import NDArray
from zcu_tools.analysis.singleshot import classify_result
from zcu_tools.experiment.utils.single_shot.ge import plot_ge_iq_density


def plot_ge_classified_iq(
    ax: Axes,
    shots: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> None:
    """Draw one preparation's 1-D raw IQ shots using the Post classifier.

    Use the fitted g/e centers and radius in the shots' IQ units. Classify by
    nearest center inside its radius and draw a blue g, red e, gray Other
    density image with the classifier overlay. The caller supplies labels and
    layout. Invalid classifier geometry raises ValueError.
    """
    g, e, other = classify_result(shots, g_center, e_center, radius)
    plot_ge_iq_density(ax, shots[g], shots[e], other=shots[other])
    plot_ge_classification_boundary(ax, g_center, e_center, radius)


def share_ge_preparation_limits(ground: Axes, excited: Axes) -> None:
    """Link two equal-aspect Post axes at the union of their plotted I/Q limits.

    Call after plotting both IQ clouds and classifier boundaries. Sharing Q as
    well as I keeps the two IQ boxes the same shape. Matplotlib raises ValueError
    if excited already shares an axis with a different peer.
    """
    g_left, g_right = ground.get_xlim()
    e_left, e_right = excited.get_xlim()
    g_bottom, g_top = ground.get_ylim()
    e_bottom, e_top = excited.get_ylim()
    excited.sharex(ground)
    excited.sharey(ground)
    ground.set_xlim(min(g_left, e_left), max(g_right, e_right))
    ground.set_ylim(min(g_bottom, e_bottom), max(g_top, e_top))


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
