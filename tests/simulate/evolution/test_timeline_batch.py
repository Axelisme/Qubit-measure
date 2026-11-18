"""Point selection preserves timeline data at the evolution boundary."""

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    DriveTrack,
    EnvelopeSpec,
    FluxSchedule,
    FluxWaveform,
    SegmentShift,
    TimelineBatch,
)


@pytest.fixture
def batch_with_continuations() -> TimelineBatch:
    ground = np.diag([1.0, 0.0])
    excited = np.diag([0.0, 1.0])
    return TimelineBatch(
        points=np.array([3, 12, 20]),
        tracks=(
            DriveTrack(
                port="drive",
                start_ns=np.array([1.0, 2.0, 4.0]),
                duration_ns=np.array([1.0, 3.0, 2.0]),
                frequency_ghz=np.array([0.1, 0.2, -0.3]),
                phase_rad=np.array([0.4, 0.5, 0.6]),
                amplitude=np.array([0.7, 0.8, -0.9]),
                envelopes=(
                    EnvelopeSpec("constant"),
                    EnvelopeSpec("sampled", np.array([1j, 1.0]), np.array([0.0, 1.0])),
                ),
                envelope_index=np.array([0, 0, 1]),
                active=np.array([True, True, False]),
            ),
        ),
        flux=FluxSchedule(
            np.array([0, 2, 3, 6]),
            np.array([0.0, 2.0, 0.0, 0.0, 3.0, 7.0]),
            np.array([[1], [2], [3], [4], [5], [6]]),
        ),
        end_ns=np.array([6.0, 8.0, 11.0]),
        flux_waveform=FluxWaveform(
            np.array([0, 3, 5, 7]),
            np.array([0.0, 2.0, 6.0, 0.0, 8.0, 3.0, 11.0]),
            np.array([[0.11], [0.12], [0.13], [0.14], [0.15], [0.16], [0.17]]),
        ),
        segment_boundaries=((2.0,), (4.0,), (7.0, 9.0)),
        segment_shifts=(
            (),
            (),
            (SegmentShift(7.0, 9.0, np.array([0.01, 0.02]), np.zeros((2, 2))),),
        ),
        start_ns=np.array([2.0, 4.0, 7.0]),
        initial_densities=np.array([ground, (ground + excited) / 2, excited]),
    )


def test_nonadjacent_points_keep_ragged_data_and_absolute_continuations(
    batch_with_continuations: TimelineBatch,
) -> None:
    selected = batch_with_continuations.select_points((2, 0))

    np.testing.assert_array_equal(selected.points, [20, 3])
    np.testing.assert_array_equal(selected.end_ns, [11.0, 6.0])
    (track,) = selected.tracks
    assert track.port == "drive"
    constant, sampled = track.envelopes
    assert constant.kind == "constant"
    assert sampled.kind == "sampled"
    np.testing.assert_array_equal(sampled.samples, [1j, 1.0])
    np.testing.assert_array_equal(sampled.sample_ns, [0.0, 1.0])
    np.testing.assert_array_equal(track.start_ns, [4.0, 1.0])
    np.testing.assert_array_equal(track.duration_ns, [2.0, 1.0])
    np.testing.assert_array_equal(track.frequency_ghz, [-0.3, 0.1])
    np.testing.assert_array_equal(track.phase_rad, [0.6, 0.4])
    np.testing.assert_array_equal(track.amplitude, [-0.9, 0.7])
    np.testing.assert_array_equal(track.envelope_index, [1, 0])
    np.testing.assert_array_equal(track.active, [False, True])
    np.testing.assert_array_equal(selected.flux.offsets, [0, 3, 5])
    np.testing.assert_array_equal(selected.flux.start_ns, [0.0, 3.0, 7.0, 0.0, 2.0])
    np.testing.assert_array_equal(selected.flux.level, [[4], [5], [6], [1], [2]])

    waveform = selected.flux_waveform
    assert waveform is not None
    np.testing.assert_array_equal(waveform.offsets, [0, 2, 5])
    np.testing.assert_array_equal(waveform.time_ns, [3.0, 11.0, 0.0, 2.0, 6.0])
    np.testing.assert_array_equal(
        waveform.flux, [[0.16], [0.17], [0.11], [0.12], [0.13]]
    )
    assert selected.segment_boundaries == ((7.0, 9.0), (2.0,))
    (shift,), no_shifts = selected.segment_shifts
    assert (shift.start_ns, shift.end_ns) == (7.0, 9.0)
    np.testing.assert_array_equal(shift.energy_shift_ghz, [0.01, 0.02])
    np.testing.assert_array_equal(shift.dephasing_per_ns, [[0.0, 0.0], [0.0, 0.0]])
    assert no_shifts == ()
    np.testing.assert_array_equal(selected.start_ns, [7.0, 2.0])
    np.testing.assert_array_equal(
        selected.initial_densities, [[[0, 0], [0, 1]], [[1, 0], [0, 0]]]
    )
