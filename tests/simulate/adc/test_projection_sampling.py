"""Projection outcomes couple distinct read events through the public ADC seam."""

import numpy as np
from numpy.testing import assert_array_equal
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate import adc


def test_raw_shots_keep_projection_outcome_correlated_with_final_state():
    assert hasattr(adc, "ProjectionReadout"), (
        "ADC must accept projection weights and event-specific initial populations"
    )
    assert hasattr(adc, "ReadoutEvent"), "ADC contents must separate logical read events"
    model = adc.ReadoutModel(
        "projection",
        ("g", "e"),
        np.array([[6.0], [6.04]]),
        6.0,
        np.array([0.04]),
        np.array([0.16]),
        np.zeros((2, 2)),
        mismatch_phase_rad=np.zeros(1),
        receiver=adc.Receiver(gain=100),
    )
    events = tuple(
        adc.ReadoutEvent(
            (adc.ReadoutPulse(start, start + 60, 1.0, 6.0),),
            (adc.SampleGrid(start + np.array([10.0, 20, 30, 40]), 6.0),),
        )
        for start in (0.0, 100.0)
    )
    batch = adc.ReadoutBatch(model, (events,), np.array([0], dtype=np.int64))
    projection = adc.ProjectionReadout(
        np.array([[0.35, 0.65]]),
        (np.eye(2)[None, ...], np.eye(2)[::-1][None, ...]),
    )
    actual = (
        adc.PreparedRawShots(projection, {3: batch}, reps=64)
        .bind(SamplingIdentity(17, (0, 0, 1)))
        .materialize(64, 0)[3]
    )
    expected = [
        np.rint(
            adc.ADC(model).response(event.pulses, event.grids[0]).traces.sum(axis=-2)
        ).astype(np.int64)
        for event in events
    ]
    ground = np.all(actual[:, 0, 0] == expected[0][0], axis=-1)
    assert 0 < ground.sum() < 64
    assert_array_equal(
        actual[ground, 0, 1], np.broadcast_to(expected[1][1], (ground.sum(), 2))
    )
    assert_array_equal(
        actual[~ground, 0, 1], np.broadcast_to(expected[1][0], ((~ground).sum(), 2))
    )
