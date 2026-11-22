"""Preparation-majority color for GE IQ clouds."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
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
