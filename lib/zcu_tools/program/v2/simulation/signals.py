"""Validate numeric signals and pack the QICK buffers without averaging."""

from collections import OrderedDict
from dataclasses import dataclass
from math import prod

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
        shape = (prod(request.loop_dims), window.reads, window.samples, 2)
        if iq.shape != shape or not np.all(np.isfinite(iq)):
            raise ValueError(
                f"invalid ADC {window.channel} signals: expected finite {shape}, got {iq.shape}"
            )
        sums = iq.sum(axis=-2)
        accumulated[window.channel] = rounded_sums(sums).reshape(-1, 2)
        decimated[window.channel] = iq.reshape(-1, 2).copy()
    return accumulated, decimated


@dataclass(frozen=True, slots=True)
class _StatisticsContent:
    conditional_shape: tuple[int, ...]
    conditional_dtype: str
    conditional_bytes: bytes
    covariance_shape: tuple[int, ...]
    covariance_dtype: str
    covariance_bytes: bytes


class StatisticsCopier:
    """Validate and detach compact rounds with a bounded, instance-local LRU.

    budget_bytes bounds retained certificates, including conservative Python
    container overhead. Zero disables reuse; negative budgets raise ValueError.
    Only successful conditional-sum and
    covariance checks are remembered, keyed by shape, dtype and exact bytes.
    Counts, request layout and noisy sums are checked on every copy. The owner
    must serialize calls, just as it serializes backend rounds.
    """

    def __init__(self, *, budget_bytes: int = 64 * 1024) -> None:
        if budget_bytes < 0:
            raise ValueError("statistics certificate budget must be nonnegative")
        self._budget_bytes = budget_bytes
        self._resident_bytes = 0
        self._certificates: OrderedDict[_StatisticsContent, int] = OrderedDict()

    def copy(self, request: AcquireRequest, result: RoundStatistics) -> RoundStatistics:
        """Validate request correspondence and detach every borrowed result array.

        Raise ValueError for channel/count/layout mismatches, nonfinite statistics,
        invalid covariance or per-shot means outside int64. No raw shots are drawn.
        Neither the input arrays nor previously returned copies are retained.
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
            raise ValueError(
                "backend state_counts do not match request point layout/reps"
            )
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
                or not np.all(np.isfinite(ch.sums))
            ):
                raise ValueError(
                    f"invalid ADC {window.channel} statistics shape or finite values"
                )
            self._validate_conditional_statistics(ch)
            rounded_sums(ch.sums / request.reps)
            channels.append(
                ChannelStatistics(
                    ch.channel,
                    ch.sums.copy(),
                    ch.conditional_sums.copy(),
                    ch.shot_covariance.copy(),
                )
            )
        return RoundStatistics(counts.copy(), tuple(channels), result.raw_identity)

    def _validate_conditional_statistics(self, ch: ChannelStatistics) -> None:
        conditional = ch.conditional_sums
        covariance = ch.shot_covariance
        # Charge keys, bytes headers, shape integers and LRU nodes conservatively.
        charge = (
            1024
            + conditional.nbytes
            + covariance.nbytes
            + 32 * (conditional.ndim + covariance.ndim)
        )
        content = None
        if charge <= self._budget_bytes:
            content = _StatisticsContent(
                conditional.shape,
                conditional.dtype.str,
                conditional.tobytes(),
                covariance.shape,
                covariance.dtype.str,
                covariance.tobytes(),
            )
            if content in self._certificates:
                self._certificates.move_to_end(content)
                return
        if not np.all(np.isfinite(conditional)) or not np.all(np.isfinite(covariance)):
            raise ValueError(
                f"invalid ADC {ch.channel} statistics shape or finite values"
            )
        if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12) or np.any(
            np.linalg.eigvalsh(covariance) < -1e-12
        ):
            raise ValueError("shot_covariance must be symmetric positive semidefinite")
        rounded_sums(conditional)
        if content is not None:
            while self._resident_bytes + charge > self._budget_bytes:
                _, evicted_charge = self._certificates.popitem(last=False)
                self._resident_bytes -= evicted_charge
            self._certificates[content] = charge
            self._resident_bytes += charge


def copy_statistics(
    request: AcquireRequest, result: RoundStatistics
) -> RoundStatistics:
    """Validate and detach a compact round without retaining validation certificates.

    Raise ValueError for channel/count/layout mismatches, nonfinite statistics,
    invalid covariance or per-shot means outside int64. No raw shots are drawn.
    """
    return StatisticsCopier(budget_bytes=0).copy(request, result)
