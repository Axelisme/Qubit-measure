"""Validate numeric signals and pack the QICK buffers without averaging."""

import numpy as np
from numpy.typing import NDArray

from zcu_tools.core.simulation import AcquireRequest, RoundSignals


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
