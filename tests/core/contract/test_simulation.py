"""Synthetic template expansion and deferred integrated-shot contracts."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core import simulation


def template_request():
    """Two structural classes with local arrays in sorted flat-point order."""
    envelope = simulation.PulseEnvelope((0.0, 0.25), (1j, -1j))
    event = simulation.EventSlot(
        gen_ch=2,
        source="drive",
        iterations=(("repeat", 1),),
        start_us=np.array([5.0, 3.0]),
        duration_us=np.array([0.5, 0.75]),
        frequency_mhz=np.array([100.0, 200.0]),
        phase_deg=np.array([10.0, 20.0]),
        gain=np.array([-0.25, 0.5]),
        envelopes=(None, envelope),
        envelope_index=np.array([0, 1], dtype=np.int64),
        active=np.array([True, False]),
        phase_reference_us=0.125,
    )
    readout = simulation.ReadoutSlot(
        channel=4,
        samples=16,
        sample_rate_mhz=8.0,
        trigger_us=np.array([[7.0, 8.0], [4.0, 6.0]]),
        frequency_mhz=np.array([[91.0, 92.0], [93.0, 94.0]]),
        phase_deg=np.array([[1.0, 2.0], [3.0, 4.0]]),
        generator_channels=np.array([[2, 3], [3, 2]], dtype=np.int64),
    )
    template = simulation.TimelineTemplate(
        points=np.array([0, 3], dtype=np.int64),
        events=(
            event,
            replace(
                event, active=np.array([True, True]), start_us=np.array([1.0, 2.0])
            ),
        ),
        readouts=(readout,),
        duration_us=np.array([10.0, 12.0]),
    )
    other = simulation.TimelineTemplate(
        points=np.array([1, 2], dtype=np.int64),
        events=(),
        readouts=(),
        duration_us=np.array([0.25, 0.5]),
    )
    return simulation.AcquireRequest(
        mode="integrated",
        loop_dims=(2, 5, 2),
        avg_level=1,
        point_dims=(2, 2),
        point_names=("flux", "frequency"),
        reps=5,
        templates=(template, other),
        controls=(),
        readouts=(
            simulation.ReadoutWindow(
                channel=4,
                gen_ch=3,
                samples=16,
                sample_rate_mhz=8.0,
                trigger_us=(0.0, 0.0),
            ),
        ),
        root_seed=42,
        sequence=0,
        rng_version="test",
    )


def test_materialize_point_selects_template_active_events_and_detaches_values():
    request = template_request()
    first = request.materialize_point(0)
    assert first.coordinates == (("flux", 0), ("frequency", 0))
    assert [event.start_us for event in first.events] == [5.0, 1.0]
    assert first.events[0].envelope is None
    last = request.materialize_point(3)
    assert last.coordinates == (("flux", 1), ("frequency", 1))
    assert last.duration_us == 12.0
    assert last.events == (
        simulation.PulseEvent(
            gen_ch=2,
            start_us=2.0,
            duration_us=0.75,
            frequency_mhz=200.0,
            phase_deg=20.0,
            gain=0.5,
            envelope=simulation.PulseEnvelope((0.0, 0.25), (1j, -1j)),
            phase_reference_us=0.125,
            source="drive",
            iterations=(("repeat", 1),),
        ),
    )
    window = last.readouts[0]
    assert window == simulation.ReadoutWindow(
        channel=4,
        gen_ch=3,
        samples=16,
        sample_rate_mhz=8.0,
        trigger_us=(4.0, 6.0),
        frequency_mhz=(93.0, 94.0),
        phase_deg=(3.0, 4.0),
        generator_channels=(3, 2),
    )
    assert request.materialize_point(2).events == ()
    assert request.materialize_point(2).duration_us == 0.5
    request.templates[0].events[1].gain[:] = 0.0
    request.templates[0].readouts[0].trigger_us[:] = 0.0
    assert last.events[0].gain == 0.5
    assert window.trigger_us == (4.0, 6.0)


@pytest.mark.parametrize("point", [-1, 4])
def test_materialize_point_rejects_out_of_range(point: int):
    with pytest.raises(ValueError, match="point"):
        template_request().materialize_point(point)


def test_templates_require_names_for_each_point_axis():
    request = template_request()
    with pytest.raises(ValueError, match="point_names"):
        replace(request, point_names=("flux",))


def round_statistics(*, noisy: bool = False):
    """Integer-separated states expose their counts across ADCs and reads."""
    counts = np.array([[[1, 3], [2, 2]], [[4, 0], [0, 4]]], dtype=np.int64)
    means = np.zeros((2, 2, 2, 2, 2), dtype=np.float64)
    means[..., 1, :] = 100.0
    means[..., 1, :, :] += 1000.0
    sums = np.sum(means * counts[..., None, :, None], axis=-2)
    covariance = np.array([[4.0, 4.0], [4.0, 4.0]]) if noisy else np.zeros((2, 2))
    return simulation.RoundStatistics(
        state_counts=counts,
        channels=(
            simulation.ChannelStatistics(
                channel=7, sums=sums, conditional_sums=means, shot_covariance=covariance
            ),
            simulation.ChannelStatistics(
                channel=2,
                sums=sums[..., :1, :] * 2,
                conditional_sums=means[..., :1, :, :] * 2,
                shot_covariance=np.zeros((2, 2)),
            ),
        ),
        raw_identity=simulation.SamplingIdentity(789, (4, 2, 1)),
    )


@pytest.mark.parametrize("avg_level", [0, 1, 2])
def test_materialize_raw_preserves_shared_state_counts_and_qick_layout(avg_level: int):
    result = round_statistics()
    raw = result.materialize_raw(reps=4, avg_level=avg_level)
    assert set(raw) == {7, 2}
    for channel in result.channels:
        shots = np.moveaxis(raw[channel.channel], avg_level, 2)
        assert shots.shape == (2, 2, 4, channel.sums.shape[-2], 2)
        assert shots.dtype == np.int64
        np.testing.assert_array_equal(shots.sum(axis=2), channel.sums)
    shots = np.moveaxis(raw[7], avg_level, 2)
    states = shots[..., 0, 0] // 100
    np.testing.assert_array_equal(
        (states == 0).sum(axis=-1), result.state_counts[..., 0]
    )
    np.testing.assert_array_equal(
        (states == 1).sum(axis=-1), result.state_counts[..., 1]
    )
    np.testing.assert_array_equal(shots[..., 1, 0] - 1000, shots[..., 0, 0])
    np.testing.assert_array_equal(raw[2][..., 0, :], raw[7][..., 0, :] * 2)


def test_materialize_raw_reproducible_noise_and_totals_within_rounding_error():
    result = round_statistics(noisy=True)
    shifted = replace(
        result,
        channels=(replace(result.channels[0], sums=result.channels[0].sums + 0.3),),
    )
    raw = shifted.materialize_raw(reps=4, avg_level=1)[7]
    np.testing.assert_array_equal(raw, shifted.materialize_raw(reps=4, avg_level=1)[7])
    assert np.all(np.abs(raw.sum(axis=1) - shifted.channels[0].sums) <= 4 / 2)
    np.testing.assert_array_equal(raw[..., 0], raw[..., 1])
    different = replace(
        shifted, raw_identity=simulation.SamplingIdentity(790, (4, 2, 1))
    )
    assert not np.array_equal(raw, different.materialize_raw(reps=4, avg_level=1)[7])
    assert not np.array_equal(raw, round_statistics().materialize_raw(4, 1)[7])
    np.testing.assert_array_equal(
        result.state_counts, [[[1, 3], [2, 2]], [[4, 0], [0, 4]]]
    )
    np.testing.assert_allclose(shifted.channels[0].sums, result.channels[0].sums + 0.3)


@pytest.mark.parametrize("reps,avg_level", [(3, 0), (4, -1), (4, 3)])
def test_materialize_raw_rejects_incompatible_layout(reps: int, avg_level: int):
    with pytest.raises(ValueError, match="reps|avg_level"):
        round_statistics().materialize_raw(reps, avg_level)


def test_materialize_raw_single_point_nearest_even_and_overflow():
    channel = simulation.ChannelStatistics(
        channel=0,
        sums=np.array([[2.5, 3.5]]),
        conditional_sums=np.array([[[2.5, 3.5]]]),
        shot_covariance=np.zeros((2, 2)),
    )
    result = simulation.RoundStatistics(
        state_counts=np.array([1], dtype=np.int64),
        channels=(channel,),
        raw_identity=simulation.SamplingIdentity(1),
    )
    np.testing.assert_array_equal(result.materialize_raw(1, 0)[0], [[[2, 4]]])
    overflowing = replace(
        result, channels=(replace(channel, sums=np.full((1, 2), 2.0**63)),)
    )
    with pytest.raises(ValueError, match="int64|overflow"):
        overflowing.materialize_raw(1, 0)


@pytest.mark.parametrize(
    "seed,components", [(-1, ()), (1, (-1,)), (True, ()), (1, (1.5,))]
)
def test_core_sampling_identity_rejects_invalid_entropy(seed, components):
    with pytest.raises(ValueError, match="nonnegative integers"):
        simulation.SamplingIdentity(seed, components)
