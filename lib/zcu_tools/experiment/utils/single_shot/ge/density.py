"""Preparation-majority color for GE IQ clouds."""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy.typing import NDArray


def plot_ge_iq_density(
    ax: Axes,
    ground: NDArray[np.complex128],
    excited: NDArray[np.complex128],
) -> AxesImage:
    """Draw two 1-D clouds of finite complex IQ shots in their input coordinates.

    The axes gain one equal-scale IQ image, returned for further styling. Each
    occupied bin is blue for a ground-preparation majority, red for an excited-
    preparation majority, or purple for a tie. Opacity rises with the log of
    the combined count; empty bins are transparent. Either cloud may be empty,
    but both empty, nonfinite shots, or non-1-D clouds raise ValueError.
    """
    if ground.ndim != 1 or excited.ndim != 1:
        raise ValueError("GE IQ clouds must be one-dimensional")
    shots = np.concatenate((ground, excited))
    if shots.size == 0 or not np.isfinite(shots).all():
        raise ValueError("GE IQ density requires finite shots")

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
    image = ax.imshow(
        rgba.transpose(1, 0, 2),
        origin="lower",
        extent=(i_edges[0], i_edges[-1], q_edges[0], q_edges[-1]),
        aspect="equal",
        interpolation="nearest",
    )
    # Shrink the axes box to keep I/Q scale exact after a constrained-layout resize.
    ax.set_adjustable("box")
    return image
