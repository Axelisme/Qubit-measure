"""Density colors for GE preparation and classification IQ clouds."""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import XAxis
from matplotlib.image import AxesImage
from matplotlib.ticker import AutoLocator, MaxNLocator
from numpy.typing import NDArray


def plot_ge_iq_density(
    ax: Axes,
    ground: NDArray[np.complex128],
    excited: NDArray[np.complex128],
    *,
    image: AxesImage | None = None,
    other: NDArray[np.complex128] | None = None,
) -> AxesImage:
    """Draw an equal-scale density image of 1-D complex IQ clouds on ax.

    Pass g/e preparations for Analyze or g/e classified shots with the optional
    Other class for Post. Use the input IQ coordinates, ignoring nonfinite shots.
    Blend blue (g), red (e) and gray (Other) in each bin using the sixth powers
    of the groups' counts, normalized to sum to one. Equal g/e counts without
    Other give purple. Pass an empty cloud for any group with no shots.
    Opacity rises with the log of the combined count; empty bins and frames
    without finite shots are transparent. Major ticks follow the IQ box size.

    To update a dedicated live IQ axes, pass its previously returned image.
    Reuse that artist and rescale from the current frame if autoscaling is on.
    Call on the plot owner thread after presenting ax; no refresh is scheduled
    here. A non-1-D cloud or a detached or foreign image raises ValueError.
    """
    clouds = (ground, excited) if other is None else (ground, excited, other)
    if any(cloud.ndim != 1 for cloud in clouds):
        raise ValueError("GE IQ clouds must be one-dimensional")
    if image is not None and image.axes is not ax:
        raise ValueError("GE IQ image must be attached to the supplied axes")
    rgba, extent = _density_frame(clouds)
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
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(_RawIQTickLocator())
    return image


class _RawIQTickLocator(AutoLocator):
    def __call__(self) -> Sequence[float]:
        axis = self.axis
        assert axis is not None
        # Budget against the drawn box, not the subplot slot. Horizontal
        # decimal labels can need more than Matplotlib's three-em estimate.
        space = axis.get_tick_space()
        bins = space // 2 if isinstance(axis, XAxis) else space
        if bins >= 9:
            return super().__call__()

        lo, hi = sorted(axis.get_view_interval())
        ticks = MaxNLocator(nbins=max(1, bins), min_n_ticks=1).tick_values(lo, hi)
        if bins > 0:
            return ticks
        # Even both end ticks can collide when there is no room for one interval.
        visible = (tick for tick in ticks if lo <= tick <= hi)
        center = 0.5 * lo + 0.5 * hi
        return [min(visible, key=lambda tick: abs(tick - center), default=center)]


def _density_frame(
    clouds: tuple[NDArray[np.complex128], ...],
) -> tuple[NDArray[np.float64], tuple[float, float, float, float]]:
    finite = tuple(cloud[np.isfinite(cloud)] for cloud in clouds)
    shots = np.concatenate(finite)
    if not shots.size:
        return np.zeros((1, 1, 4)), (0, 1, 0, 1)

    # All groups vote on the same grid, including constant I or Q.
    i_edges = np.histogram_bin_edges(shots.real, bins=90)
    q_edges = np.histogram_bin_edges(shots.imag, bins=90)
    counts = np.stack(
        [
            np.histogram2d(cloud.real, cloud.imag, bins=(i_edges, q_edges))[0]
            for cloud in finite
        ]
    )
    total = counts.sum(axis=0)
    # Normalizing the sixth powers cancels the shared per-bin total.
    weights = counts**6
    weight_sum = weights.sum(axis=0)
    red = np.divide(
        weights[1], weight_sum, out=np.full_like(total, 0.5), where=total > 0
    )
    gray = np.zeros_like(total)
    if len(clouds) == 3:
        np.divide(0.5 * weights[2], weight_sum, out=gray, where=total > 0)
    opacity = np.where(
        total > 0, 0.15 + 0.85 * np.log1p(total) / np.log1p(total.max()), 0.0
    )
    rgba = np.stack((red + gray, gray, 1.0 - red - gray, opacity), axis=-1)
    return rgba.transpose(1, 0, 2), (i_edges[0], i_edges[-1], q_edges[0], q_edges[-1])
