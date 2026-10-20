"""Shared Floquet phase-sampling and probe-certificate bin policies."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

_MIN_PHASE_SAMPLES = 128
_SAMPLES_PER_RATIO = 32
MAX_PHASE_SAMPLES = 4096
MAX_SAMPLING_RATIO = MAX_PHASE_SAMPLES / _SAMPLES_PER_RATIO
PROBE_BIN_WIDTH_GHZ = 0.002
PROBE_BIN_HALF_WIDTH_GHZ = PROBE_BIN_WIDTH_GHZ / 2


def phase_sample_count(
    span_ghz: ArrayLike, frequency_ghz: ArrayLike
) -> NDArray[np.float64]:
    """Return uncapped sample counts for finite spans and positive frequencies.

    span_ghz includes the drive. Inputs broadcast; scalar inputs return a
    zero-dimensional array. Multiply span by the sampling factor before division
    to preserve the original rounding. Builders reject counts above
    MAX_PHASE_SAMPLES; cost estimates cap them. Guards compare span/frequency
    with MAX_SAMPLING_RATIO before building, without log2 boundary rounding.
    """
    scaled = _SAMPLES_PER_RATIO * np.asarray(span_ghz) / np.asarray(frequency_ghz)
    return np.asarray(
        2 ** np.ceil(np.log2(np.maximum(_MIN_PHASE_SAMPLES, scaled))),
        dtype=np.float64,
    )


def probe_bin_index(frequency_ghz: ArrayLike) -> NDArray[np.float64]:
    """Return nearest-even 2 MHz bin indices with the input shape.

    Frequencies are signed GHz values. Divide before NumPy rounding, including
    for negative values and half-bin boundaries; do not replace this with floor.
    """
    return np.asarray(
        np.round(np.asarray(frequency_ghz) / PROBE_BIN_WIDTH_GHZ), dtype=np.float64
    )


def probe_bin_center(frequency_ghz: ArrayLike) -> NDArray[np.float64]:
    """Return signed GHz certificate centers with the input frequency shape."""
    return np.asarray(
        probe_bin_index(frequency_ghz) * PROBE_BIN_WIDTH_GHZ, dtype=np.float64
    )
