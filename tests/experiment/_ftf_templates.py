"""Synthetic FTF template builders and result narrowing shared by composition tests."""

from dataclasses import replace

import numpy as np
from zcu_tools.core.simulation import (
    AcquireRequest,
    AcquisitionMode,
    EnvironmentSnapshot,
    EventSlot,
    PulseEnvelope,
    ReadoutSlot,
    ReadoutWindow,
    RoundSignals,
    RoundStatistics,
    TimelineTemplate,
)
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


def template(*, tones: int = 0) -> TimelineTemplate:
    """Two points; controls stop at 4 ns, readout starts at 8/10 ns."""

    def slot(channel: int, start, duration, frequency, phase, gain) -> EventSlot:
        return EventSlot(
            channel,
            "synthetic",
            (),
            np.array(start, dtype=float),
            np.full(2, duration),
            np.full(2, frequency),
            np.full(2, phase),
            np.full(2, gain),
            (None,),
            np.zeros(2, dtype=np.int64),
            np.ones(2, dtype=bool),
            0.001,
        )

    drives = tuple(
        slot(2, [0.0, 0.002], 0.002, 250.0 + i * 100, 90.0, -0.2) for i in range(tones)
    )
    ro = slot(0, [0.008, 0.010], 0.020, 6000.0, 30.0, 0.1)
    ro = replace(ro, envelopes=(PulseEnvelope((0.0, 0.020), (2j, 2j)),))
    readout = ReadoutSlot(
        0,
        4,
        1000.0,
        np.array([[0.008], [0.010]]),
        np.full((2, 1), 6000.0),
        np.full((2, 1), 45.0),
        np.zeros((2, 1), dtype=np.int64),
    )
    return TimelineTemplate(
        np.array([0, 1], dtype=np.int64), (*drives, ro), (readout,), np.full(2, 0.030)
    )


def settings():
    value = parse_ftf(ftf_document())
    return value.model_copy(
        update={
            "numerics": value.numerics.model_copy(
                update={
                    "truncation": value.numerics.truncation.model_copy(
                        update={"retained_levels": 2}
                    )
                }
            )
        }
    )


def snapshot(value: float = 0.2) -> EnvironmentSnapshot:
    return EnvironmentSnapshot("ftf-v2", "synthetic", (("bias", value),))


def request(mode: AcquisitionMode = "integrated", *, templates=None) -> AcquireRequest:
    return AcquireRequest(
        mode,
        (3, 2),
        0,
        (ReadoutWindow(0, 0, 4, 1000.0, (0.008,)),),
        123,
        7,
        f"PCG64/SeedSequence-v2/numpy-{np.__version__}",
        point_dims=(2,),
        point_names=("sweep",),
        reps=3,
        templates=(template(),) if templates is None else templates,
    )


def expect_statistics(result: RoundStatistics | RoundSignals) -> RoundStatistics:
    """Narrow the integrated result once for typed numerical assertions."""
    if not isinstance(result, RoundStatistics):
        raise AssertionError("integrated FTF round must return RoundStatistics")
    return result
