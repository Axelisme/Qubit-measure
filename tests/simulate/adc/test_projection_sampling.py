"""Projection outcomes couple distinct read events through the public ADC seam."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from zcu_tools.core.simulation import SamplingIdentity
from zcu_tools.simulate import adc


def projection_inputs():
    assert hasattr(adc, "ProjectionReadout"), (
        "ADC must accept projection weights and event-specific initial populations"
    )
    assert hasattr(adc, "ReadoutEvent"), (
        "ADC contents must separate logical read events"
    )
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
    return model, events, batch, projection


def subset_inputs():
    """ADC 3 observes both events; ADC 1 observes only the terminal event."""
    model, events, batch, projection = projection_inputs()
    rates = np.array([[0.0, 0.03], [0.02, 0.0]])
    batch = replace(batch, model=replace(model, rates_per_ns=rates))
    terminal_only = replace(batch, events=((replace(events[0], grids=()), events[1]),))
    return {3: batch, 1: terminal_only}, projection


def test_decimated_adc_without_herald_keeps_terminal_event_path_and_read_count():
    batches, projection = subset_inputs()
    actual = adc.sample_traces(
        projection, batches, reps=64, identity=SamplingIdentity(29, (0, 0, 1))
    )
    both, terminal = actual.channels
    assert both.iq.shape == (64, 2, 4, 2)
    assert terminal.iq.shape == (64, 1, 4, 2)
    assert_array_equal(terminal.iq[:, 0], both.iq[:, 1])


def test_integrated_adc_without_herald_uses_terminal_state_not_herald_outcome():
    batches, projection = subset_inputs()
    channels = tuple(
        adc.ChannelReadout(
            channel,
            adc.ADC(batch.model).responses(batch),
            batch.content_index,
            batch.model.receiver,
        )
        for channel, batch in batches.items()
    )
    assert channels[0].conditional.event_indices == (0, 1)
    assert channels[1].conditional.event_indices == (1,)
    actual = adc.round_statistics(
        projection, channels, reps=1000, seed=29, acquire_sequence=0, round_index=0
    )
    both, terminal = actual.channels
    assert both.sums.shape == (1, 2, 2)
    assert terminal.sums.shape == (1, 1, 2)
    assert_array_equal(
        terminal.conditional_sums[..., 0, :, :], both.conditional_sums[..., 1, :, :]
    )
    # The missing herald cannot re-label the remaining read as event zero.
    assert (
        np.max(
            np.abs(
                terminal.conditional_sums[..., 0, :, :]
                - both.conditional_sums[..., 0, :, :]
            )
        )
        > 1
    )


def test_raw_adc_without_herald_keeps_terminal_event_path_and_read_count():
    batches, projection = subset_inputs()
    actual = (
        adc.PreparedRawShots(projection, batches, reps=128)
        .bind(SamplingIdentity(29, (0, 0, 1)))
        .materialize(128, 0)
    )
    assert actual[3].shape == (128, 1, 2, 2)
    assert actual[1].shape == (128, 1, 1, 2)
    assert_array_equal(actual[1][..., 0, :], actual[3][..., 1, :])
    assert np.unique(actual[1][..., 0, :].reshape(-1, 2), axis=0).shape[0] > 8


def test_raw_shots_keep_projection_outcome_correlated_with_final_state():
    model, events, batch, projection = projection_inputs()
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


def test_conditional_responses_preserve_ordered_event_windows():
    model, events, batch, _ = projection_inputs()
    actual = adc.ADC(model).responses(batch)
    expected = np.stack(
        [
            adc.ADC(model).response(event.pulses, event.grids[0]).traces.sum(axis=-2)
            for event in events
        ]
    )
    np.testing.assert_allclose(actual.sums[0], expected, rtol=1e-8, atol=1e-8)
    assert actual.event_indices == (0, 1)


def test_integrated_joint_counts_condition_early_reads_on_outcome():
    assert hasattr(adc.ConditionalSums, "event_indices"), (
        "integrated conditional sums must identify each read's logical event"
    )
    _, _, _, projection = projection_inputs()
    conditional = adc.ConditionalSums(
        np.array([[[[10.0, 0.0], [20.0, 0.0]], [[30.0, 0.0], [40.0, 0.0]]]]),
        np.array([4, 4], dtype=np.int64),
        event_indices=(0, 1),
    )
    channel = adc.ChannelReadout(
        3, conditional, np.array([0], dtype=np.int64), adc.Receiver()
    )
    actual = adc.round_statistics(
        projection, (channel,), reps=1000, seed=19, acquire_sequence=0, round_index=0
    )
    counts = actual.state_counts
    assert counts.shape == (1, 4)
    assert_array_equal(counts[0, [0, 3]], [0, 0])
    assert abs(counts[0, 1] / 1000 - 0.35) < 0.05
    np.testing.assert_allclose(
        actual.channels[0].sums[0],
        [
            [10 * counts[0, 1] + 20 * counts[0, 2], 0],
            [40 * counts[0, 1] + 30 * counts[0, 2], 0],
        ],
        rtol=0,
        atol=40,
    )
    assert_array_equal(
        actual.channels[0].conditional_sums,
        np.array(
            [
                [
                    [[10.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 0.0]],
                    [[30.0, 0.0], [40.0, 0.0], [30.0, 0.0], [40.0, 0.0]],
                ]
            ]
        ),
    )


def test_events_draw_independent_ctmc_but_share_paths_across_adc_windows():
    model, events, _, _ = projection_inputs()
    model = replace(model, rates_per_ns=np.array([[0.0, 0.03], [0.02, 0.0]]))
    events = tuple(replace(event, grids=event.grids * 2) for event in events)
    batch = adc.ReadoutBatch(model, (events,), np.array([0], dtype=np.int64))
    readout = adc.ProjectionReadout(np.ones((1, 1)), (np.array([[[1.0, 0.0]]]),) * 2)
    actual = (
        adc.PreparedRawShots(readout, {3: batch, 1: batch}, reps=128)
        .bind(SamplingIdentity(23, (0, 0, 1)))
        .materialize(128, 0)
    )
    assert_array_equal(actual[3], actual[1])
    assert_array_equal(actual[3][..., 0, :], actual[3][..., 1, :])
    assert_array_equal(actual[3][..., 2, :], actual[3][..., 3, :])
    # Equal deterministic starts do not couple random jumps of separate events.
    assert (
        np.count_nonzero(np.any(actual[3][..., 0, :] != actual[3][..., 2, :], axis=-1))
        > 32
    )


@pytest.mark.parametrize(
    "weights,populations",
    [
        (np.array([0.3, 0.6]), (np.eye(2),)),
        (np.array([0.3, 0.7]), (np.array([[1.0, 0.0], [-0.1, 1.1]]),)),
        (np.array([0.3, 0.7]), (np.ones((1, 2)),)),
        (np.array([0.3, 0.7]), (np.eye(2), np.ones((2, 1)))),
        (np.array([0.3, 0.7]), ()),
    ],
)
def test_projection_rejects_invalid_probabilities_or_event_layout(weights, populations):
    with pytest.raises(ValueError, match="population|event"):
        adc.ProjectionReadout(weights, populations)


def test_projection_owns_inputs_and_preserves_tolerated_probability_residuals():
    weights = np.array([0.35, 0.6500001])
    populations = np.eye(2)
    readout = adc.ProjectionReadout(weights, (populations,))
    weights[:] = 0
    populations[:] = 0
    assert_array_equal(readout.weights, [0.35, 0.6500001])
    assert_array_equal(readout.populations[0], np.eye(2))
    assert not readout.weights.flags.writeable
    assert not readout.populations[0].flags.writeable


def test_decimated_events_keep_projection_outcomes_correlated():
    model, events, batch, projection = projection_inputs()
    actual = (
        adc.sample_traces(
            projection, {3: batch}, reps=64, identity=SamplingIdentity(17, (0, 0, 1))
        )
        .channels[0]
        .iq
    )
    expected = [
        adc.ADC(model).response(event.pulses, event.grids[0]).traces for event in events
    ]
    ground = np.all(np.isclose(actual[:, 0], expected[0][0]), axis=(-2, -1))
    assert 0 < ground.sum() < 64
    np.testing.assert_allclose(
        actual[ground, 1], np.broadcast_to(expected[1][1], actual[ground, 1].shape)
    )
    np.testing.assert_allclose(
        actual[~ground, 1], np.broadcast_to(expected[1][0], actual[~ground, 1].shape)
    )
