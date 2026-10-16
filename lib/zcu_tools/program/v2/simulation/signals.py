"""Validate numeric signals and pack the QICK buffers without averaging."""

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import (
    AcquireRequest,
    ChannelStatistics,
    RoundSignals,
    RoundStatistics,
)


def rounded_sums(values: NDArray[np.float64]) -> NDArray[np.int64]:
    rounded = np.rint(values)
    # float64 represents 2**63, but not INT64_MAX; the upper bound is exclusive.
    if (
        not np.all(np.isfinite(rounded))
        or np.any(rounded < -(2.0**63))
        or np.any(rounded >= 2.0**63)
    ):
        raise ValueError("ADC sums must be finite and representable as int64")
    return rounded.astype(np.int64)


def pack_signals(
    request: AcquireRequest, signals: RoundSignals
) -> tuple[dict[int, NDArray[np.int64]], dict[int, NDArray[np.float64]]]:
    by_channel = {signal.channel: signal.iq for signal in signals.channels}
    if len(by_channel) != len(signals.channels) or set(by_channel) != {
        w.channel for w in request.readouts
    }:
        raise ValueError("backend ADC channels do not match request")
    accumulated = {}
    decimated = {}
    for window in request.readouts:
        iq = np.asarray(by_channel[window.channel], dtype=np.float64)
        shape = (request.executions, window.reads)
        shape += (window.samples, 2) if request.mode == "decimated" else (2,)
        if iq.shape != shape or not np.all(np.isfinite(iq)):
            raise ValueError(
                f"invalid ADC {window.channel} signals: expected finite {shape}, got {iq.shape}"
            )
        sums = iq.sum(axis=-2) if request.mode == "decimated" else iq
        accumulated[window.channel] = rounded_sums(sums).reshape(-1, 2)
        if request.mode == "decimated":
            decimated[window.channel] = iq.reshape(-1, 2).copy()
    return accumulated, decimated


def copy_statistics(
    request: AcquireRequest, result: RoundStatistics
) -> RoundStatistics:
    """Validate request correspondence and detach a compact round's borrowed arrays.

    Raise ValueError for channel/count/layout mismatches, nonfinite statistics,
    invalid covariance or per-shot means outside int64. No raw shots are drawn.
    """
    point_dims = tuple(
        n for axis, n in enumerate(request.loop_dims) if axis != request.avg_level
    )
    counts = result.state_counts
    if (
        counts.ndim != len(point_dims) + 1
        or counts.shape[:-1] != point_dims
        or counts.shape[-1] < 1
        or counts.dtype != np.int64
        or np.any(counts < 0)
        or np.any(counts.sum(axis=-1) != request.reps)
    ):
        raise ValueError("backend state_counts do not match request point layout/reps")
    by_channel = {ch.channel: ch for ch in result.channels}
    if len(by_channel) != len(result.channels) or set(by_channel) != {
        w.channel for w in request.readouts
    }:
        raise ValueError("backend ADC channels do not match request")
    channels = []
    for window in request.readouts:
        ch = by_channel[window.channel]
        shape = (*point_dims, window.reads, 2)
        conditional_shape = (*point_dims, window.reads, counts.shape[-1], 2)
        if (
            ch.sums.shape != shape
            or ch.conditional_sums.shape != conditional_shape
            or ch.shot_covariance.shape != (2, 2)
            or not all(
                np.all(np.isfinite(x))
                for x in (ch.sums, ch.conditional_sums, ch.shot_covariance)
            )
        ):
            raise ValueError(
                f"invalid ADC {window.channel} statistics shape or finite values"
            )
        covariance = ch.shot_covariance
        if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12) or np.any(
            np.linalg.eigvalsh(covariance) < -1e-12
        ):
            raise ValueError("shot_covariance must be symmetric positive semidefinite")
        rounded_sums(ch.conditional_sums)
        rounded_sums(ch.sums / request.reps)
        channels.append(
            ChannelStatistics(
                ch.channel,
                ch.sums.copy(),
                ch.conditional_sums.copy(),
                covariance.copy(),
            )
        )
    return RoundStatistics(counts.copy(), tuple(channels), result.raw_identity)
