"""FTF playback preserves valid one-sample envelopes in template playback."""

import numpy as np
import pytest
from zcu_tools.core.simulation import (
    EnvironmentSnapshot,
    PointTimeline,
    PulseEnvelope,
    PulseEvent,
    ReadoutWindow,
)
from zcu_tools.experiment.ftf import plan_template
from zcu_tools.resources.ftf import parse_ftf

from tests.core._simulation import TIME_BASE, single_point_template, tick_tuple, ticks
from tests.resources._ftf_data import ftf_document

pytestmark = pytest.mark.sim


def test_single_signed_complex_sample_keeps_full_plateau_and_identity():
    settings = parse_ftf(ftf_document())
    pulse = PulseEvent(
        2,
        ticks(0.0),
        ticks(0.008),
        0.0,
        0.0,
        0.2,
        PulseEnvelope(ticks(0.001), (-0.4 + 0.3j,)),
    )
    readout = PulseEvent(0, ticks(0.008), ticks(0.002), 6000.0, 0.0, 0.1)
    window = ReadoutWindow(
        0,
        0,
        1,
        TIME_BASE.period_ticks(1000.0),
        tick_tuple((0.008,)),
        (6000.0,),
        (0.0,),
        (0,),
    )
    point = plan_template(
        settings,
        EnvironmentSnapshot("ftf-v2", "test", (("bias", 0.2),)),
        single_point_template(
            PointTimeline(
                (), (pulse, readout), (window,), ticks(0.010), time_base=TIME_BASE
            )
        ),
        time_base=TIME_BASE,
    )
    track = point.batch.tracks[0]
    envelope = track.envelopes[0]
    assert envelope.identity
    np.testing.assert_allclose(
        track.amplitude[0] * envelope.values(np.array([0.0, 1.5, 8.0])),
        [-0.08 + 0.06j] * 3,
        atol=1e-15,
        rtol=0,
    )
