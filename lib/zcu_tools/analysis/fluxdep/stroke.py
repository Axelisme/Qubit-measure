"""Normalized polyline brushes shared by grid-mask and point-cloud callers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, hypot, isfinite

import numpy as np
from numpy.typing import NDArray

from .selection import points_in_normalized_brush, toggle_near_mask


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
    before modifying mask. Squared width, axis spans, normalized coordinates
    and segment distances must also remain finite. A complete stroke may use at
    most 10,000 samples including segment endpoints; validate the full sampling
    budget before painting. Axes and stroke are not modified.
    """
    x_span = _axis_span(dev_values, "dev_values")
    y_span = _axis_span(freqs, "freqs")
    if mask.shape != (dev_values.size, freqs.size):
        raise ValueError("mask shape must match (len(dev_values), len(freqs))")
    samples = _stroke_samples(stroke, width, x_span, y_span)
    _validate_normalized_coordinates(dev_values, freqs, samples, x_span, y_span)

    for point in samples:
        toggle_near_mask(dev_values, freqs, mask, point.x, point.y, width, select)


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
    nonfinite data/bounds/vertices, zero or nonfinite spans, empty stroke or
    invalid width. Squared width, normalized coordinates and segment distances
    must also be finite. A complete stroke may use at most 10,000 samples including
    segment endpoints; validate its complete budget before evaluating membership.
    """
    if xs.shape != ys.shape:
        raise ValueError("xs and ys must have the same shape")
    if xs.ndim != 1:
        raise ValueError("xs and ys must be 1D arrays")
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError("xs and ys must be finite")
    x_span = _bound_span(x_bound, "x_bound")
    y_span = _bound_span(y_bound, "y_bound")
    samples = _stroke_samples(stroke, width, x_span, y_span)
    _validate_normalized_coordinates(xs, ys, samples, x_span, y_span)

    membership = np.zeros(xs.shape, dtype=np.bool_)
    for point in samples:
        membership |= points_in_normalized_brush(
            xs,
            ys,
            x=point.x,
            y=point.y,
            width=width,
            x_bound=x_bound,
            y_bound=y_bound,
        )
    return membership


def _bound_span(bound: tuple[float, float], name: str) -> float:
    """Validate finite distinct endpoints and return their absolute span."""
    if not all(isfinite(value) for value in bound):
        raise ValueError(f"{name} endpoints must be finite")
    span = abs(bound[1] - bound[0])
    if span == 0 or not isfinite(span):
        raise ValueError(f"{name} span must be finite and non-zero")
    return span


def _axis_span(axis: NDArray[np.float64], name: str) -> float:
    """Validate a finite 1D grid axis and return its absolute endpoint span."""
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} axis must be 1D with at least two entries")
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} axis must be finite")
    return _bound_span((float(axis[0]), float(axis[-1])), name)


def _validate_stroke(stroke: Sequence[BrushPoint], width: float) -> None:
    """Reject invalid vertices or radius before any circular brush is applied."""
    if not isfinite(width) or width < 0 or not isfinite(width * width):
        raise ValueError(
            "width must be non-negative with finite width and squared width"
        )
    if not stroke:
        raise ValueError("stroke must be nonempty")
    for point in stroke:
        if not isfinite(point.x) or not isfinite(point.y):
            raise ValueError("stroke vertices must be finite")
    if width == 0 and any(point != stroke[0] for point in stroke):
        raise ValueError("width must be positive for a moving stroke")


def _stroke_samples(
    stroke: Sequence[BrushPoint], width: float, x_span: float, y_span: float
) -> list[BrushPoint]:
    """Validate and sample the full stroke within the 10,000-sample budget."""
    _validate_stroke(stroke, width)
    sample_limit = 10_000
    if len(stroke) > sample_limit:
        raise ValueError("stroke sample budget exceeds 10,000")
    samples = [stroke[0]]
    for start, end in zip(stroke[:-1], stroke[1:], strict=True):
        distance = hypot((end.x - start.x) / x_span, (end.y - start.y) / y_span)
        if not isfinite(distance):
            raise ValueError("stroke segment distance must be finite")
        intervals = 1
        if distance > 0:
            required_intervals = distance / width * 2
            if not isfinite(required_intervals) or required_intervals > sample_limit:
                raise ValueError("stroke sample budget exceeds 10,000")
            intervals = max(1, ceil(required_intervals))
        if len(samples) + intervals > sample_limit:
            raise ValueError("stroke sample budget exceeds 10,000")
        for index in range(1, intervals):
            fraction = index / intervals
            samples.append(
                BrushPoint(
                    start.x + (end.x - start.x) * fraction,
                    start.y + (end.y - start.y) * fraction,
                )
            )
        samples.append(end)
    return samples


def _validate_normalized_coordinates(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    samples: Sequence[BrushPoint],
    x_span: float,
    y_span: float,
) -> None:
    """Check all circle-coordinate arithmetic before evaluating any brush."""
    # Overflow is translated into a request error, never a partially painted mask.
    with np.errstate(over="ignore", invalid="ignore"):
        for point in samples:
            if not isfinite(point.x) or not isfinite(point.y):
                raise ValueError("stroke sample coordinates must be finite")
            if (
                not np.isfinite((xs - point.x) / x_span).all()
                or not np.isfinite((ys - point.y) / y_span).all()
            ):
                raise ValueError("normalized stroke coordinates must be finite")
