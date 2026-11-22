"""Preparation-majority color for GE IQ clouds."""

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from numpy.typing import NDArray


def plot_ge_iq_density(
    ax: Axes,
    ground: NDArray[np.complex128],
    excited: NDArray[np.complex128],
    *,
    image: AxesImage | None = None,
) -> AxesImage:
    """Draw one equal-scale image of two 1-D complex IQ clouds on ax.

    Use the input IQ coordinates, ignoring nonfinite shots. Each occupied bin
    is blue for a ground-preparation majority, red for an excited-preparation
    majority, or purple for a tie. Opacity rises with the log of the combined
    count; empty bins and frames without finite shots are transparent.

    To update a dedicated live IQ axes, pass its previously returned image.
    Reuse that artist and rescale from the current frame if autoscaling is on.
    Call on the plot owner thread after presenting ax; no refresh is scheduled
    here. A non-1-D cloud or a detached or foreign image raises ValueError.
    """
    if ground.ndim != 1 or excited.ndim != 1:
        raise ValueError("GE IQ clouds must be one-dimensional")
    if image is not None and image.axes is not ax:
        raise ValueError("GE IQ image must be attached to the supplied axes")
    rgba, extent = _density_frame(ground, excited)
    if image is not None:
        image.set_data(rgba)
        image.set_extent(extent)
        # set_extent grows dataLim. Recompute so a tighter frame can shrink it.
        ax.relim()
        ax.autoscale_view()
        return image

    image = ax.imshow(
        rgba, origin="lower", extent=extent, aspect="equal", interpolation="nearest"
    )
    # Shrink the axes box to keep I/Q scale exact after a constrained-layout resize.
    ax.set_adjustable("box")
    return image


def plot_ge_classification_boundary(
    ax: Axes, g_center: complex, e_center: complex, radius: float
) -> tuple[Line2D, Line2D, Line2D]:
    """Overlay the GE radius classifier on an autoscaled raw IQ axes.

    Supply the distinct finite g/e centers and nonnegative finite radius used
    for classification, in the same units as the IQ image. Draw both centers
    and the radius boundaries; overlapping disks share one bisector segment.
    Extend the auto limits to include the boundaries with a margin. Return
    representative g-center, e-center and boundary handles for the caller's
    legend. This only draws the geometry; it does not classify the IQ data.
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
    boundary = _plot_region_edge(ax, ground)
    _plot_region_edge(ax, excited)
    if cut > 0:
        # The ends of the ground arc are the two circle intersections.
        _plot_region_edge(ax, ground[[0, -1]])

    # Keep the centers distinguishable when the fitted radius spans only a few pixels.
    (g_marker,) = ax.plot(
        g_center.real,
        g_center.imag,
        "o",
        color="0.15",
        markerfacecolor="white",
        markersize=2.5,
        markeredgewidth=0.7,
    )
    (e_marker,) = ax.plot(
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
    return g_marker, e_marker, boundary


def _plot_region_edge(ax: Axes, samples: NDArray[np.complex128]) -> Line2D:
    (edge,) = ax.plot(
        samples.real,
        samples.imag,
        "--",
        color="0.15",
        linewidth=0.9,
        path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")],
    )
    return edge


def _density_frame(
    ground: NDArray[np.complex128], excited: NDArray[np.complex128]
) -> tuple[NDArray[np.float64], tuple[float, float, float, float]]:
    ground, excited = ground[np.isfinite(ground)], excited[np.isfinite(excited)]
    shots = np.concatenate((ground, excited))
    if not shots.size:
        return np.zeros((1, 1, 4)), (0, 1, 0, 1)

    # Both preparations must vote on the same grid, including constant I or Q.
    i_edges = np.histogram_bin_edges(shots.real, bins=90)
    q_edges = np.histogram_bin_edges(shots.imag, bins=90)
    g, _, _ = np.histogram2d(ground.real, ground.imag, bins=(i_edges, q_edges))
    e, _, _ = np.histogram2d(excited.real, excited.imag, bins=(i_edges, q_edges))
    total = g + e
    red = np.where(e == g, 0.5, e > g)
    opacity = np.where(
        total > 0, 0.15 + 0.85 * np.log1p(total) / np.log1p(total.max()), 0.0
    )
    rgba = np.stack((red, np.zeros_like(red), 1.0 - red, opacity), axis=-1)
    return rgba.transpose(1, 0, 2), (i_edges[0], i_edges[-1], q_edges[0], q_edges[-1])
