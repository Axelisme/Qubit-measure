"""Numeric ADC-sum conversion shared by environments and program transport."""

import numpy as np
from numpy.typing import NDArray


def rounded_sums(values: NDArray[np.float64]) -> NDArray[np.int64]:
    """Own nearest-even int64 sample sums, preserving the input shape.

    values is a float64 array in unnormalized ADC sample-sum units. Inputs are
    never mutated. Nonfinite values and rounded values outside int64 raise
    ValueError rather than clipping or wrapping. Each call owns fresh storage.
    """
    rounded = np.rint(values)
    # float64 represents 2**63, but not INT64_MAX; upper bound is exclusive.
    if (
        not np.all(np.isfinite(rounded))
        or np.any(rounded < -(2.0**63))
        or np.any(rounded >= 2.0**63)
    ):
        raise ValueError("ADC sums must be finite and representable as int64")
    return rounded.astype(np.int64)


def shot_layout(
    values: NDArray[np.float64], point_dims: tuple[int, ...], reps: int, avg_level: int
) -> NDArray[np.int64]:
    """Round point-major shots and insert the QICK repetition axis.

    values has shape (prod(point_dims), reps, reads, 2). point_dims are C-order
    sweep dimensions; reps is a nonnegative integer; avg_level is an integer in
    [0,len(point_dims)]. Return owned int64 (*loop_dims,reads,2), where loop_dims
    inserts reps at avg_level. Invalid layouts or unsafe sums raise ValueError.
    """
    if type(reps) is not int or reps < 0:
        raise ValueError("reps must be a nonnegative integer")
    if type(avg_level) is not int or not 0 <= avg_level <= len(point_dims):
        raise ValueError("avg_level is outside the point-axis layout")
    if (
        values.ndim != 4
        or values.shape[:2] != (int(np.prod(point_dims)), reps)
        or values.shape[-1] != 2
    ):
        raise ValueError("shot layout must be (points,reps,reads,2)")
    rounded = rounded_sums(values)
    layout = rounded.reshape((*point_dims, reps, values.shape[-2], 2))
    return np.moveaxis(layout, -3, avg_level).copy()
