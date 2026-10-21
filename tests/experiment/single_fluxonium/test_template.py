"""Public compiled-template planning, without evolving or constructing an ADC."""

import importlib
from dataclasses import replace

import numpy as np
import pytest

from ._inputs import settings, snapshot, template


def test_plan_preserves_complete_timeline_and_filtered_flux() -> None:
    planner = importlib.import_module("zcu_tools.experiment.single_fluxonium.template")
    plan = planner.plan_template(settings(), snapshot(), template())
    batch = plan.batch
    np.testing.assert_array_equal(batch.points, [0, 3])
    np.testing.assert_array_equal(batch.end_ns, [16, 16])
    assert len(batch.tracks) == 1
    track = batch.tracks[0]
    assert track.port == "mw"
    np.testing.assert_array_equal(track.start_ns, [2, 2])
    np.testing.assert_array_equal(track.duration_ns, [6, 6])
    np.testing.assert_array_equal(track.frequency_ghz, [0.25, 0.25])
    np.testing.assert_allclose(track.phase_rad, [np.pi / 2, np.pi / 2])
    np.testing.assert_array_equal(track.amplitude, [-0.2, -0.2])
    assert batch.flux.level.shape[1] == 1
    np.testing.assert_array_equal(batch.flux.level[batch.flux.offsets[:-1]], [[0], [0]])
    waveform = batch.flux_waveform
    assert waveform is not None
    times = waveform.time_ns[:waveform.offsets[1]]
    values = waveform.flux[:waveform.offsets[1], 0]
    # Analytic first-order step: delayed turn-on at 2 ns, switch-off at 10 ns.
    assert np.interp(4, times, values) == pytest.approx(0.5 + 0.1 * (1 - np.exp(-1)))
    assert np.interp(16, times, values) == pytest.approx(
        0.5 + 0.1 * (1 - np.exp(-4)) * np.exp(-3)
    )
    assert plan.readout.pulses[0][0].start_ns == 16
    assert plan.readout.pulses[0][0].amplitude == 1
    np.testing.assert_array_equal(plan.readout.grids[0][0].times_ns, [18, 19, 20, 21])
    # Detached plan arrays do not change when borrowed template storage changes.
    original = template()
    detached = planner.plan_template(settings(), snapshot(), original)
    original.events[0].gain[:] = 0.9
    np.testing.assert_array_equal(detached.batch.tracks[0].amplitude, [-0.2, -0.2])
