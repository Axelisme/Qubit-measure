"""Synthetic single-point template inputs for numeric contract tests."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from zcu_tools.core.simulation import (
    EventSlot,
    PointTimeline,
    ReadoutSlot,
    TimeBase,
    TimelineTemplate,
)

TIME_BASE = TimeBase.from_clocks((1000.0, 430.08, 76.8))


def ticks(time_us: float) -> int:
    """Return exact synthetic fixture ticks, rejecting off-grid microseconds."""
    return TIME_BASE.ticks(time_us)


def tick_array(times_us: ArrayLike) -> NDArray[np.int64]:
    """Convert finite fixture microseconds exactly; retain the input shape."""
    times = np.asarray(times_us, dtype=np.float64)
    return np.array([ticks(float(t)) for t in times.flat], dtype=np.int64).reshape(
        times.shape
    )


def tick_tuple(times_us: Iterable[float]) -> tuple[int, ...]:
    """Return exact trigger ticks from fixture microseconds in emission order."""
    return tuple(ticks(t) for t in times_us)


def single_point_template(timeline: PointTimeline) -> TimelineTemplate:
    """Represent detached diagnostic controls as a one-point template fixture."""
    events = tuple(
        EventSlot(
            event.gen_ch,
            event.source,
            event.iterations,
            np.array([event.start_ticks]),
            np.array([event.duration_ticks]),
            np.array([event.frequency_mhz]),
            np.array([event.phase_deg]),
            np.array([event.gain]),
            (event.envelope,),
            np.zeros(1, dtype=np.int64),
            np.ones(1, dtype=bool),
            event.phase_reference_ticks,
        )
        for event in timeline.events
    )
    readouts = tuple(
        ReadoutSlot(
            window.channel,
            window.samples,
            window.sample_period_ticks,
            np.array([window.trigger_ticks]),
            np.array([window.frequency_mhz or (0.0,) * window.reads]),
            np.array([window.phase_deg or (0.0,) * window.reads]),
            np.array(
                [window.generator_channels or (window.gen_ch,) * window.reads],
                dtype=np.int64,
            ),
        )
        for window in timeline.readouts
    )
    return TimelineTemplate(
        np.array([0], dtype=np.int64),
        events,
        readouts,
        np.array([timeline.duration_ticks]),
    )
