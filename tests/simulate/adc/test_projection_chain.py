"""Markov-chain projection inputs at the public ADC boundaries."""

from collections import Counter
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from zcu_tools.simulate import adc


def chain_statistics_inputs(point_dims):
    initial = np.broadcast_to([0.4, 0.6], (*point_dims, 2))
    transitions = tuple(
        np.broadcast_to(values, (*point_dims, 2, 2))
        for values in ([[0.8, 0.2], [0.3, 0.7]], [[0.1, 0.9], [0.6, 0.4]])
    )
    chain = adc.ProjectionChain(initial, transitions)
    indicator = np.zeros((1, 3, 2, 2))
    indicator[0, :, 1, 0] = 1
    all_events = adc.ChannelReadout(
        3,
        adc.ConditionalSums(
            indicator, np.ones(3, dtype=np.int64), event_indices=(0, 1, 2)
        ),
        np.zeros(point_dims, dtype=np.int64),
        adc.Receiver(),
    )
    contents = np.array(
        [
            [[[10.0, 1], [20, 2]], [[30, 3], [40, 4]]],
            [[[50.0, 5], [60, 6]], [[70, 7], [80, 8]]],
        ]
    )
    subset = adc.ChannelReadout(
        1,
        adc.ConditionalSums(
            contents,
            np.ones(2, dtype=np.int64),
            event_indices=(0, 2),
            output_order=np.array([[1, 0], [0, 1]], dtype=np.int64),
        ),
        np.arange(np.prod(point_dims), dtype=np.int64).reshape(point_dims) % 2,
        adc.Receiver(noise_covariance=np.array([[0.25, 0], [0, 0.5]])),
    )
    return chain, (all_events, subset)


def test_chain_integrated_occupancies_match_enumerated_two_shot_joint_law():
    chain, channels = chain_statistics_inputs((8000,))
    actual = adc.round_statistics(
        chain, channels, reps=2, seed=91, acquire_sequence=0, round_index=0
    )
    # Enumerate real shot paths and their independent two-shot products, not flow counts.
    paths = list(product(range(2), repeat=3))
    probabilities = [
        chain.initial[0, path[0]]
        * chain.transitions[0][0, path[0], path[1]]
        * chain.transitions[1][0, path[1], path[2]]
        for path in paths
    ]
    expected = Counter()
    for left, right in product(range(len(paths)), repeat=2):
        occupancy = tuple(a + b for a, b in zip(paths[left], paths[right], strict=True))
        expected[occupancy] += probabilities[left] * probabilities[right]
    noiseless = np.einsum(
        "...b,...rbi->...ri", actual.state_counts, actual.channels[0].conditional_sums
    )
    observed = Counter(map(tuple, noiseless[..., 0].astype(int)))
    # Fixed-seed sampling of 8000 independent points; TV catches lost cross-event correlation.
    tv = (
        sum(
            abs(observed[key] / 8000 - probability)
            for key, probability in expected.items()
        )
        / 2
    )
    assert tv < 0.04
    assert_array_equal(actual.state_counts.sum(axis=-1), np.full(8000, 2))
    assert actual.state_counts.shape[-1] <= 4
    # Distinct content permutations and nonconsecutive logical events must align on bins.
    states = actual.channels[0].conditional_sums[..., 0]
    contents = channels[1].content_index
    expected_subset = np.empty_like(actual.channels[1].conditional_sums)
    for point, content in enumerate(contents):
        for column, read in enumerate(channels[1].conditional.column_order[content]):
            event = channels[1].conditional.event_indices[read]
            expected_subset[point, column] = channels[1].conditional.sums[
                content, read, states[point, event].astype(int)
            ]
    assert_array_equal(actual.channels[1].conditional_sums, expected_subset)
    assert_allclose(actual.channels[1].shot_covariance, [[0.25, 0], [0, 0.5]])
    assert_allclose(actual.channels[0].sums, noiseless, atol=2.5, rtol=0)


@pytest.mark.parametrize("reps", [0, 100000])
def test_chain_prepared_statistics_keep_bounded_categories_and_multiaxis_points(reps):
    chain, channels = chain_statistics_inputs((2, 3))
    prepared = adc.PreparedStatistics(chain, channels, reps=reps)
    actual = prepared.draw(seed=73, acquire_sequence=2, round_index=1)
    assert actual.state_counts.shape == (2, 3, 4)
    assert_array_equal(actual.state_counts.sum(axis=-1), np.full((2, 3), reps))
    for channel in actual.channels:
        assert channel.conditional_sums.shape[-2] == 4
        if reps == 0:
            assert_array_equal(channel.sums, np.zeros_like(channel.sums))
    repeated = prepared.draw(seed=73, acquire_sequence=2, round_index=1)
    assert_array_equal(actual.state_counts, repeated.state_counts)
    for left, right in zip(actual.channels, repeated.channels, strict=True):
        assert_array_equal(left.sums, right.sums)


def test_chain_owns_validated_rows_without_probability_repair():
    assert hasattr(adc, "ProjectionChain"), "ADC needs a compact Markov-chain input"
    initial = np.array([[0.35, 0.6500001]])
    transition = np.array([[[1.0000001, 0.0], [0.25, 0.75]]])
    chain = adc.ProjectionChain(initial, (transition,))
    initial[:] = 0
    transition[:] = 0
    assert_array_equal(chain.initial, [[0.35, 0.6500001]])
    assert_array_equal(chain.transitions[0], [[[1.0000001, 0], [0.25, 0.75]]])
    assert not chain.initial.flags.writeable
    assert not chain.transitions[0].flags.writeable
    single = adc.ProjectionChain(np.array([0.3, 0.7]), ())
    assert_array_equal(single.initial, [0.3, 0.7])


@pytest.mark.parametrize(
    "initial,transitions",
    [
        (np.array([]), ()),
        (np.array([0.3, 0.6]), ()),
        (np.array([-0.1, 1.1]), ()),
        (np.array([np.nan, 1.0]), ()),
        (np.array([0.3, 0.7]), (np.ones((2, 1)),)),
        (np.array([[0.3, 0.7]]), (np.eye(2),)),
        (np.array([0.3, 0.7]), (np.array([[0.3, 0.6], [0, 1.0]]),)),
        (np.array([0.3, 0.7]), (np.array([[-0.1, 1.1], [0, 1.0]]),)),
    ],
)
def test_chain_rejects_invalid_probabilities_and_transition_shapes(
    initial, transitions
):
    assert hasattr(adc, "ProjectionChain"), "ADC needs a compact Markov-chain input"
    with pytest.raises(ValueError, match="population|transition"):
        adc.ProjectionChain(initial, transitions)
