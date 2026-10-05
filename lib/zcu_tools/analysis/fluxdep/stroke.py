"""Normalized polyline brushes shared by grid-mask and point-cloud callers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BrushPoint:
    """One finite stroke vertex in the caller's native data-axis units.

    x is a device/flux coordinate; y is a frequency coordinate. Their units
    match the supplied axes or point arrays, without conversion.
    """

    x: float
    y: float


def apply_mask_stroke(
    dev_values: NDArray[np.float64],
    freqs: NDArray[np.float64],
    mask: NDArray[np.bool_],
    stroke: Sequence[BrushPoint],
    width: float,
    *,
    select: bool,
) -> None:
    """Select or erase a polyline brush in a grid mask, in place.

    Axes are finite 1D arrays with at least two entries and distinct endpoints.
    mask has shape (len(dev_values), len(freqs)). stroke is nonempty and finite,
    in native axis units. width is a finite nonnegative normalized radius:
    each axis is scaled by its absolute endpoint span. Samples include vertices
    and have normalized separation at most width / 2. Each sample applies the
    existing circular mask brush. select=True adds; False erases.

    A zero radius is valid only for a stationary stroke. Out-of-bounds vertices
    are allowed; only supplied grid positions can be changed. ValueError rejects
    invalid shapes, nonfinite data, zero spans, an empty stroke, or invalid width
    before modifying mask. Axes and stroke are not modified.
    """
    raise NotImplementedError("normalized mask stroke is not implemented")


def points_in_normalized_stroke(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    *,
    stroke: Sequence[BrushPoint],
    width: float,
    x_bound: tuple[float, float],
    y_bound: tuple[float, float],
) -> NDArray[np.bool_]:
    """Return point membership in a normalized polyline brush, without mutation.

    xs and ys are same-shaped finite 1D arrays in native axis units. stroke is
    a nonempty finite sequence in those units. Each finite bound pair has distinct
    endpoints (either order). width is a finite nonnegative normalized radius.
    Include vertices and sample segments at normalized separation <= width / 2;
    return the union of the existing circular brush memberships for all samples.

    A zero radius is valid only for a stationary stroke. Vertices outside bounds
    remain valid; bounds define scaling, not clipping. An empty point cloud returns
    a zero-length bool array for a valid request. ValueError rejects invalid shapes,
    nonfinite data/bounds/vertices, zero spans, empty stroke or invalid width.
    """
    raise NotImplementedError("normalized point stroke is not implemented")
