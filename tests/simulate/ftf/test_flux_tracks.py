"""Batch filtering agrees with the independent scalar convolution contract."""

import numpy as np
import pytest
from zcu_tools.simulate import ftf
from zcu_tools.simulate.evolution import (
    ControlPulse,
    ControlSequence,
    Envelope,
    EnvelopeSpec,
)
from zcu_tools.simulate.lines import FilteredFlux, FluxLine


@pytest.mark.parametrize("gain", [0.7, -0.7])
@pytest.mark.parametrize("sampled", [False, True])
def test_batch_midpoints_and_waveform_match_scalar_filter(gain, sampled):
    spec = (
        EnvelopeSpec(
            "sampled",
            sample_ns=np.array([0.0, 4.0, 10.0]),
            samples=np.array([0.2 + 0.1j, 1 - 0.3j, 0.4 + 0.2j]),
        )
        if sampled
        else EnvelopeSpec("constant")
    )
    track = ftf.FluxTrack(
        "bias",
        np.array([4.0, 8.0, 0.0]),
        np.array([12.0, 16.0, 0.0]),
        np.array([0.023, -0.017, 0.0]),
        np.array([0.2, -0.4, 0.0]),
        np.array([gain, gain / 2, 0.0]),
        (spec,),
        np.zeros(3, dtype=np.int64),
        np.array([True, True, False]),
    )
    dc = np.array([0.31, 0.17, 0.43])
    lines = (FluxLine("bias", (0.11, -0.06, 0.03), 3.0, 6.0),)
    schedule = ftf.schedule(
        (track,), dc_flux=dc, lines=lines, sample_period_ns=2.0, points=3
    )
    waveform = ftf.flux_waveform(
        (track,), dc_flux=dc, lines=lines, sample_period_ns=2.0, points=3
    )
    for p in range(2):
        start, duration = track.start_ns[p], track.duration_ns[p]
        # FilteredFlux uses local carrier phase; FluxTrack uses absolute phase.
        envelope = (
            Envelope.flat()
            if not sampled
            else Envelope.sampled(spec.sample_ns, spec.samples, outside="hold")
        )
        control = ControlPulse(
            "bias",
            start,
            start + duration,
            Envelope(
                lambda t, gain=track.amplitude[p], envelope=envelope: (
                    gain * envelope.at(t)
                )
            ),
            track.frequency_ghz[p],
            track.phase_rad[p] + 2 * np.pi * track.frequency_ghz[p] * start,
            breakpoints_ns=tuple(start + t for t in spec.sample_ns if t <= duration),
        )
        reference = FilteredFlux(dc, ControlSequence((control,)), lines)
        lo, hi = waveform.offsets[p : p + 2]
        times = waveform.time_ns[lo:hi]
        assert np.all(np.diff(times) == 2)
        np.testing.assert_allclose(
            waveform.flux[lo : hi - 1],
            [reference.at(t) for t in times[:-1]],
            atol=2e-11,
            rtol=0,
        )
        np.testing.assert_array_equal(waveform.flux[hi - 1], dc)
        assert np.max(np.abs(reference.at(times[-1]) - dc)) < 5e-4
        lo, hi = schedule.offsets[p : p + 2]
        starts = schedule.start_ns[lo:hi]
        levels = schedule.level[lo:hi]
        assert starts[0] == 0
        assert np.any(levels != 0)
        np.testing.assert_array_equal(levels[-1], [0, 0, 0])
        for t in times[:-1]:
            index = np.searchsorted(starts, t, side="right") - 1
            expected = np.rint((reference.at(t + 1) - dc) / 1e-3).astype(int)
            np.testing.assert_array_equal(levels[index], expected)
    lo, hi = schedule.offsets[2:4]
    np.testing.assert_array_equal(schedule.start_ns[lo:hi], [0])
    np.testing.assert_array_equal(schedule.level[lo:hi], [[0, 0, 0]])


def test_overlapping_lines_sum_before_quantizing_and_tails_use_absolute_bounds():
    track = ftf.FluxTrack(
        "slow",
        np.array([0.0]),
        np.array([12.0]),
        np.zeros(1),
        np.zeros(1),
        np.array([-1.0]),
        (EnvelopeSpec("constant"),),
        np.zeros(1, dtype=np.int64),
        np.ones(1, dtype=bool),
    )
    other = ftf.FluxTrack(
        "fast",
        np.array([0.0]),
        np.array([12.0]),
        np.zeros(1),
        np.zeros(1),
        np.array([1.0]),
        (EnvelopeSpec("constant"),),
        np.zeros(1, dtype=np.int64),
        np.ones(1, dtype=bool),
    )
    lines = (
        FluxLine("slow", (0.1, 0, 0), tau_ns=20),
        FluxLine("fast", (0.1, 0, 0), tau_ns=2),
    )
    result = ftf.schedule(
        (track, other), dc_flux=np.zeros(3), lines=lines, sample_period_ns=2, points=1
    )
    controls = ControlSequence(
        (ControlPulse("slow", 0, 12, Envelope.flat(-1)), ControlPulse("fast", 0, 12))
    )
    reference = FilteredFlux(np.zeros(3), controls, lines)
    assert result.start_ns[-1] > 12
    for t in np.arange(0.0, result.start_ns[-1], 2.0):
        index = np.searchsorted(result.start_ns, t, side="right") - 1
        np.testing.assert_array_equal(
            result.level[index], np.rint(reference.at(t + 1) / 1e-3)
        )


def test_empty_template_has_one_dc_span_per_point():
    dc_flux = np.array([0.2, 0.3, 0.4])
    schedule = ftf.schedule((), dc_flux=dc_flux, lines=(), sample_period_ns=2, points=4)
    waveform = ftf.flux_waveform(
        (), dc_flux=dc_flux, lines=(), sample_period_ns=2, points=4
    )
    np.testing.assert_array_equal(schedule.offsets, np.arange(5))
    np.testing.assert_array_equal(schedule.level, np.zeros((4, 3), dtype=int))
    np.testing.assert_array_equal(waveform.offsets, np.arange(5))
    np.testing.assert_array_equal(waveform.flux, np.tile(dc_flux, (4, 1)))


def test_track_validation_rejects_invalid_shape_and_unknown_source():
    with pytest.raises(ValueError, match="shape"):
        ftf.FluxTrack(
            "bias",
            np.zeros(2),
            np.zeros(1),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            (EnvelopeSpec("constant"),),
            np.zeros(2, dtype=np.int64),
            np.ones(2, dtype=bool),
        )
    track = ftf.FluxTrack(
        "unknown",
        np.zeros(1),
        np.ones(1) * 4,
        np.zeros(1),
        np.zeros(1),
        np.ones(1),
        (EnvelopeSpec("constant"),),
        np.zeros(1, dtype=np.int64),
        np.ones(1, dtype=bool),
    )
    with pytest.raises(ValueError, match="source"):
        ftf.schedule(
            (track,), dc_flux=np.zeros(3), lines=(), sample_period_ns=2, points=1
        )
