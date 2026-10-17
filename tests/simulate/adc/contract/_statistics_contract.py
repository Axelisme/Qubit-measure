"""Reusable black-box suite: subclass and provide a round_statistics_impl fixture.

The fixture returns any callable implementing the public round_statistics signature.
No test depends on a solver or on the implementation's private sampling helpers.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.core.simulation import RoundStatistics, SamplingIdentity
from zcu_tools.simulate import adc


class StatisticsImplementation(Protocol):
    """Callable seam accepted by the distribution contract suite."""

    def __call__(
        self,
        populations: NDArray[np.float64],
        channels: tuple[adc.ChannelReadout, ...],
        *,
        reps: int,
        seed: int,
        acquire_sequence: int,
        round_index: int,
    ) -> RoundStatistics: ...


def scenario(
    points: int = 24000,
) -> tuple[NDArray[np.float64], tuple[adc.ChannelReadout, ...]]:
    """Two contents, two reads and two ADCs sharing a nontrivial state mixture."""
    populations = np.broadcast_to([0.25, 0.75], (2, points // 2, 2)).copy()
    means = np.array([[[[0.0, 0.0], [4.0, -2.0]], [[0.0, 0.0], [4.0, -2.0]]]])
    means = np.concatenate((means, means + 10), axis=0)
    conditional = adc.ConditionalSums(means, np.array([4, 4], dtype=np.int64))
    index = np.zeros(populations.shape[:-1], dtype=np.int64)
    index[1] = 1
    receiver = adc.Receiver(noise_covariance=np.array([[0.5, 0.1], [0.1, 0.25]]))
    return populations, (
        adc.ChannelReadout(3, conditional, index, receiver),
        adc.ChannelReadout(1, conditional, index, receiver),
    )


class RoundStatisticsContract:
    """Distribution and stream promises for any batched statistics implementation."""

    @pytest.mark.parametrize("reps", [0, 17])
    def test_counts_sum_to_reps_and_are_shared_by_channels(
        self,
        round_statistics_impl: StatisticsImplementation,
        reps: int,
    ) -> None:
        populations, channels = scenario(12)
        result = round_statistics_impl(
            populations, channels, reps=reps, seed=37, acquire_sequence=2, round_index=5
        )
        assert result.state_counts.shape == (2, 6, 2)
        assert result.state_counts.dtype == np.int64
        assert np.all(result.state_counts >= 0)
        np.testing.assert_array_equal(result.state_counts.sum(axis=-1), reps)
        assert {c.channel for c in result.channels} == {1, 3}
        for channel in result.channels:
            assert channel.sums.shape == (2, 6, 2, 2)
            assert channel.conditional_sums.shape == (2, 6, 2, 2, 2)
            if reps == 0:
                np.testing.assert_array_equal(channel.sums, 0)
            np.testing.assert_allclose(channel.shot_covariance, [[2, 0.4], [0.4, 1]])

    def test_sums_have_mixture_mean_and_receiver_plus_rounding_covariance(
        self,
        round_statistics_impl: StatisticsImplementation,
    ) -> None:
        populations, channels = scenario()
        result = round_statistics_impl(
            populations, channels, reps=16, seed=37, acquire_sequence=2, round_index=5
        )
        for channel in result.channels:
            # Worked two-state mixture: E[mu]=(3,-1.5), Var(mu)=(3,.75),
            # Cov(I,Q)=-1.5. Four samples add receiver covariance (2,.4; .4,1).
            centered = channel.sums.copy()
            centered[1] -= 160  # second content adds ten ADC units per shot
            samples = centered.reshape(-1, 2, 2)[:, 0]
            np.testing.assert_allclose(
                samples.mean(axis=0), [48, -24], atol=0.25, rtol=0
            )
            np.testing.assert_allclose(
                np.cov(samples.T),
                [[81.3333333333, -17.6], [-17.6, 29.3333333333]],
                atol=2.0,
                rtol=0,
            )
            # Conditioning on the actual shared counts removes mixture variance.
            predicted = np.einsum(
                "...s,...rsi->...ri", result.state_counts, channel.conditional_sums
            )
            residual = (channel.sums - predicted).reshape(-1, 2, 2)[:, 0]
            np.testing.assert_allclose(
                np.cov(residual.T),
                [[33.3333333333, 6.4], [6.4, 17.3333333333]],
                atol=0.9,
                rtol=0,
            )

    def test_zero_receiver_still_has_reps_over_twelve_rounding_variance(
        self,
        round_statistics_impl: StatisticsImplementation,
    ) -> None:
        populations, _ = scenario()
        conditional = adc.ConditionalSums(
            np.zeros((1, 1, 2, 2)), np.array([4], dtype=np.int64)
        )
        channels = (
            adc.ChannelReadout(
                0,
                conditional,
                np.zeros(populations.shape[:-1], dtype=np.int64),
                adc.Receiver(),
            ),
        )
        result = round_statistics_impl(
            populations, channels, reps=16, seed=37, acquire_sequence=2, round_index=5
        )
        samples = result.channels[0].sums.reshape(-1, 2)
        np.testing.assert_allclose(samples.mean(axis=0), 0, atol=0.03, rtol=0)
        np.testing.assert_allclose(samples.var(axis=0), 4 / 3, atol=0.06, rtol=0)

    def test_statistics_use_one_round_stream_and_raw_uses_separate_identity(
        self,
        round_statistics_impl: StatisticsImplementation,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        populations, channels = scenario(12)
        seed_sequence = np.random.SeedSequence
        entropies: list[object] = []

        def record_seed_sequence(entropy=None, *args, **kwargs):
            # Delegate at the external RNG boundary; do not replace the draw.
            entropies.append(entropy)
            return seed_sequence(entropy, *args, **kwargs)

        monkeypatch.setattr(np.random, "SeedSequence", record_seed_sequence)
        result = round_statistics_impl(
            populations, channels, reps=17, seed=37, acquire_sequence=2, round_index=5
        )
        assert entropies == [(37, 2, 5, 0)]
        assert result.raw_identity == SamplingIdentity(37, (2, 5, 1))
