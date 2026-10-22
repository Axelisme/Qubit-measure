"""Bounded marginal histogram composition and independently retained raw shots."""

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    AcquireRequest,
    ChannelStatistics,
    RoundStatistics,
    SamplingIdentity,
)
from zcu_tools.program.v2.simulation import IntegratedRound, RawShots

from .routing import RoutedGroup


@dataclass(frozen=True)
class CavityRound:
    """group: original-point/cavity ownership; result: compact ADC delivery.

    All ADCs in result share this cavity's sampled populations and histories.
    Different CavityRounds at the same point are independent physical systems.
    """

    group: RoutedGroup
    result: IntegratedRound


@dataclass(frozen=True)
class IndependentRawShots:
    """Retained compact raw sources with original loop-layout ownership.

    point_dims: C-order non-repetition shape. reps: fixed acquire shots.
    pieces: (original flat point indices, detached raw source) pairs. Each source
    supplies all ADCs assigned to its cavity; no source borrows backend buffers.
    """

    point_dims: tuple[int, ...]
    reps: int
    pieces: tuple[tuple[NDArray[np.int64], RawShots], ...]

    def materialize(self, reps: int, avg_level: int) -> dict[int, NDArray[np.int64]]:
        """Return fresh per-ADC int64 arrays in the original acquire loop order.

        Wrong reps or avg_level outside 0..len(point_dims) raises ValueError.
        Child ADC sources own physical sampling and overflow checks. No backend
        execution, device reads or shared RNG advancement occurs here.
        """
        if reps != self.reps or not 0 <= avg_level <= len(self.point_dims):
            raise ValueError("raw reps/layout do not match the acquire")
        output: dict[int, NDArray[np.int64]] = {}
        for points, source in self.pieces:
            for channel, values in source.materialize(reps, 0).items():
                if channel not in output:
                    output[channel] = np.empty(
                        (reps, prod(self.point_dims), *values.shape[-2:]),
                        dtype=np.int64,
                    )
                output[channel][:, points] = values.reshape(
                    reps, len(points), *values.shape[-2:]
                )
        return {
            channel: np.moveaxis(
                values.reshape(reps, *self.point_dims, *values.shape[-2:]), 0, avg_level
            )
            for channel, values in output.items()
        }


def combine_integrated(
    request: AcquireRequest,
    rounds: tuple[CavityRound, ...],
    levels: tuple[int, ...],
    index: int,
) -> IntegratedRound:
    """Combine independent cavity marginals without enumerating joint states.

    rounds partition requested ADCs/points. Each cavity's sampled multinomial
    counts are preserved exactly. Sorted marginal cumulative-count boundaries
    form a sparse common histogram with at most 1+sum(levels-1) bins per point;
    zero-width duplicate boundaries are retained. Conditional responses select
    each marginal's state on these bins. This is a representation of integrated
    marginal moments, not a joint state distribution or raw-shot sampler.
    Raw shots retain the independent cavity sources, including shared histories
    for multiple ADCs observing the same cavity at the same point.
    """
    points = prod(request.point_dims)
    counts = [np.zeros((points, size), dtype=np.int64) for size in levels]
    for value in counts:
        value[:, 0] = request.reps
    for piece in rounds:
        counts[piece.group.cavity][piece.group.points] = (
            piece.result.statistics.state_counts
        )
    cumulative = [np.cumsum(value, axis=-1) for value in counts]
    boundaries = np.sort(
        np.concatenate(
            [
                np.zeros((points, 1), dtype=np.int64),
                *[value[:, :-1] for value in cumulative],
                np.full((points, 1), request.reps, dtype=np.int64),
            ],
            axis=-1,
        ),
        axis=-1,
    )
    joint = np.diff(boundaries, axis=-1)
    bins = joint.shape[-1]
    channels: dict[int, ChannelStatistics] = {}
    for piece in rounds:
        group = piece.group
        states = np.sum(
            boundaries[group.points, :-1, None]
            >= cumulative[group.cavity][group.points, None, :-1],
            axis=-1,
        )
        states = np.minimum(states, levels[group.cavity] - 1)
        for channel in piece.result.statistics.channels:
            if channel.channel not in channels:
                reads = channel.sums.shape[-2]
                channels[channel.channel] = ChannelStatistics(
                    channel.channel,
                    np.empty((points, reads, 2)),
                    np.empty((points, reads, bins, 2)),
                    channel.shot_covariance,
                )
            destination = channels[channel.channel]
            destination.sums[group.points] = channel.sums
            destination.conditional_sums[group.points] = np.take_along_axis(
                channel.conditional_sums, states[:, None, :, None], axis=-2
            )
    shaped = tuple(
        ChannelStatistics(
            channel.channel,
            channel.sums.reshape(*request.point_dims, *channel.sums.shape[1:]),
            channel.conditional_sums.reshape(
                *request.point_dims, *channel.conditional_sums.shape[1:]
            ),
            channel.shot_covariance,
        )
        for channel in channels.values()
    )
    statistics = RoundStatistics(
        joint.reshape(*request.point_dims, bins),
        shaped,
        SamplingIdentity(request.root_seed, (request.sequence, index, 1)),
    )
    return IntegratedRound(
        statistics,
        IndependentRawShots(
            request.point_dims,
            request.reps,
            tuple((piece.group.points.copy(), piece.result.raw) for piece in rounds),
        ),
    )
