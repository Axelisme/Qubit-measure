"""Synthetic single-point template inputs for numeric contract tests."""

import numpy as np
from zcu_tools.core.simulation import (
    EventSlot,
    PointTimeline,
    ReadoutSlot,
    TimelineTemplate,
)


def single_point_template(timeline: PointTimeline) -> TimelineTemplate:
    """Represent detached diagnostic controls as a one-point template fixture."""
    events = tuple(
        EventSlot(
            event.gen_ch,
            event.source,
            event.iterations,
            np.array([event.start_us]),
            np.array([event.duration_us]),
            np.array([event.frequency_mhz]),
            np.array([event.phase_deg]),
            np.array([event.gain]),
            (event.envelope,),
            np.zeros(1, dtype=np.int64),
            np.ones(1, dtype=bool),
            event.phase_reference_us,
        )
        for event in timeline.events
    )
    readouts = tuple(
        ReadoutSlot(
            window.channel,
            window.samples,
            window.sample_rate_mhz,
            np.array([window.trigger_us]),
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
        np.array([timeline.duration_us]),
    )
