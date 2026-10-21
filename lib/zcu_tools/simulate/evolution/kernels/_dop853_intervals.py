"""Knot-aligned local clocks for the shared density DOP853 chain.

Subtracting a row origin can collapse adjacent source knots into one float.
A compensated high/low difference retains their ordering and bridge duration.
Each playback clock then recovers the original source endpoint, rather than
rounding a row-local difference back across an event.
"""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray


class Interval(NamedTuple):
    """One smooth integration span with ns endpoints in each playback clock.

    duration: positive ns integrated from local zero. row_start/end preserve
    carrier and interaction-picture time relative to the original row.
    flux_start/end are absolute waveform times. envelope is (tones,2), with
    pulse-local endpoints for each active tone. Endpoints retain source knots;
    no time epsilon or zero-duration event replacement is used.
    """

    duration: float
    row_start: float
    row_end: float
    flux_start: float
    flux_end: float
    envelope: NDArray[np.float64]


@njit(nogil=True, cache=True)
def _difference(time, origin):
    """Return an error-free high/low representation of time minus origin."""
    high = time - origin
    virtual_origin = time - high
    low = (time - (high + virtual_origin)) + (virtual_origin - origin)
    return high, low


@njit(nogil=True, cache=True)
def _source_time(origin, parts):
    """Add an origin to a compensated offset, rounding only the final sum."""
    high, low = parts
    total = origin + high
    virtual_high = total - origin
    error = (origin - (total - virtual_high)) + (high - virtual_high)
    return total + (error + low)


@njit(nogil=True, cache=True)
def _append_knots(knots, times, origin, duration):
    # Include neighboring source samples before comparing compensated offsets:
    # origin + duration can itself round onto either side of a boundary.
    first = max(0, np.searchsorted(times, origin, side="left") - 1)
    stop = min(len(times), np.searchsorted(times, origin + duration, side="right") + 1)
    for index in range(first, stop):
        parts = _difference(times[index], origin)
        if (0.0, 0.0) < parts < (duration, 0.0):
            knots.append(parts)


@njit(nogil=True, cache=True)
def row_intervals(segments, row):
    """Partition one row at all active envelope and packed flux sample knots.

    Returns increasing positive-duration Interval records. Only exactly equal
    compensated offsets merge. A zero-duration row returns no intervals.
    Flux samples outside the row and inactive envelope windows add no stops.
    """
    duration = float(segments.duration[row])
    knots = [(0.0, 0.0), (duration, 0.0)]
    for tone in range(segments.tones[row]):
        envelope = segments.envelope[row, tone]
        begin, end = segments.sample_offsets[envelope : envelope + 2]
        _append_knots(
            knots,
            segments.sample_time[begin:end],
            segments.offset[row, tone],
            duration,
        )
    if len(segments.flux_offsets):
        point = segments.timeline[row]
        begin, end = segments.flux_offsets[point : point + 2]
        _append_knots(
            knots, segments.flux_time[begin:end], segments.start[row], duration
        )
    knots.sort()
    intervals = []
    previous = knots[0]
    for current in knots[1:]:
        if current == previous:
            continue
        length = (current[0] - previous[0]) + (current[1] - previous[1])
        envelope = np.empty((segments.tones[row], 2))
        for tone in range(segments.tones[row]):
            origin = segments.offset[row, tone]
            envelope[tone, 0] = _source_time(origin, previous)
            envelope[tone, 1] = _source_time(origin, current)
        intervals.append(
            Interval(
                length,
                _source_time(0.0, previous),
                _source_time(0.0, current),
                _source_time(segments.start[row], previous),
                _source_time(segments.start[row], current),
                envelope,
            )
        )
        previous = current
    return intervals


@njit(nogil=True, cache=True)
def playback_time(time: float, start: float, end: float, duration: float) -> float:
    """Map interval-local ns to a source clock, retaining both exact endpoints.

    Interior stages can round onto an endpoint, but never across it. Local zero
    and duration return the stored source coordinates without subtraction and
    readdition. The source clock may be constant across an unresolvable bridge.
    """
    if time <= 0:
        return start
    if time >= duration:
        return end
    return min(end, max(start, start + time))
