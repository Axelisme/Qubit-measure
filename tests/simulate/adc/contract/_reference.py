"""Small test-only sampler validating that the reusable contract is satisfiable."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from zcu_tools.core.simulation import (
    ChannelStatistics,
    RoundStatistics,
    SamplingIdentity,
)
from zcu_tools.simulate import adc


def reference_statistics(
    populations: NDArray[np.float64],
    channels: tuple[adc.ChannelReadout, ...],
    *,
    reps: int,
    seed: int,
    acquire_sequence: int,
    round_index: int,
) -> RoundStatistics:
    """Draw each point independently, sharing counts across every ADC and read."""
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence((seed, acquire_sequence, round_index, 0))
        )
    )
    counts = np.array(
        [
            rng.multinomial(reps, p)
            for p in populations.reshape(-1, populations.shape[-1])
        ],
        dtype=np.int64,
    ).reshape(populations.shape)
    output = []
    for channel in channels:
        conditional = channel.conditional.sums[channel.content_index]
        covariance = (
            int(channel.conditional.sample_counts[0])
            * channel.receiver.noise_covariance
        )
        means = np.einsum("...s,...rsi->...ri", counts, conditional)
        noise = rng.multivariate_normal(
            np.zeros(2), reps * (covariance + np.eye(2) / 12), size=means.shape[:-1]
        )
        output.append(
            ChannelStatistics(channel.channel, means + noise, conditional, covariance)
        )
    return RoundStatistics(
        counts,
        tuple(output),
        SamplingIdentity(seed, (acquire_sequence, round_index, 1)),
    )
