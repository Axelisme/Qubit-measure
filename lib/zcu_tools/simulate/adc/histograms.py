"""Common cumulative-boundary encoding of integrated marginal histograms."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HistogramBins:
    """Owned category partition preserving each supplied marginal histogram.

    counts: int64 (*points, bins) nonnegative category widths, summing to the
        shared histogram total per point. Duplicate boundaries give zero bins.
    indices: tuple in input-histogram order, each int64 (*points, bins), selecting
        that histogram's category for each common bin. These selections are
        a marginal-moment encoding, not a joint state distribution or shot path.
    """

    counts: NDArray[np.int64]
    indices: tuple[NDArray[np.int64], ...]


def cumulative_count_bins(histograms: tuple[NDArray[np.int64], ...]) -> HistogramBins:
    """Encode marginal counts on common sorted cumulative-count boundaries.

    histograms is a nonempty tuple of nonnegative int64 (*points, categories)
    arrays, with identical point shapes, positive category counts and equal
    totals per point. Totals may be zero or differ between points. Inputs are
    not mutated. Invalid dtype, counts, shapes or totals raise ValueError.

    Return owned counts and per-histogram category indices on exactly
    1+sum(categories-1) bins. Keep repeated boundaries as zero-count bins; the
    width is shared by every point and independent of the number of shots.
    This preserves each histogram exactly, not a joint state distribution.
    """
    if not histograms:
        raise ValueError("at least one histogram is required")
    first = np.asarray(histograms[0])
    if first.ndim < 1:
        raise ValueError("histograms must have a category axis")
    cumulative = []
    for histogram in histograms:
        values = np.asarray(histogram)
        if (
            values.dtype != np.int64
            or values.ndim < 1
            or values.shape[:-1] != first.shape[:-1]
            or values.shape[-1] == 0
            or np.any(values < 0)
        ):
            raise ValueError(
                "histograms must be nonnegative int64 with matching points"
            )
        cumulative.append(np.cumsum(values, axis=-1))
    total = cumulative[0][..., -1:]
    if any(np.any(values[..., -1:] != total) for values in cumulative):
        raise ValueError("histogram totals must agree per point")
    boundaries = np.sort(
        np.concatenate(
            [np.zeros_like(total), *(values[..., :-1] for values in cumulative), total],
            axis=-1,
        ),
        axis=-1,
    )
    return HistogramBins(
        np.diff(boundaries, axis=-1),
        tuple(
            np.sum(boundaries[..., :-1, None] >= values[..., None, :-1], axis=-1)
            for values in cumulative
        ),
    )
