"""Stateless NPZ waveform samples and opaque JSON recipe format.

No path I/O, repository lookup, or formula interpretation is performed here.
Consumers read bytes and pass them to decode_npz. Resources alone interprets
formula recipes; program and simulation play the saved samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

MAX_ARB_WAVEFORM_SAMPLES = 1_000_001

WaveformErrorReason = Literal[
    "array_not_numeric",
    "array_not_1d",
    "array_non_finite",
    "array_shape_mismatch",
    "sample_count_too_small",
    "sample_count_too_large",
    "time_must_start_at_zero",
    "time_not_strictly_increasing",
    "amplitude_out_of_range",
    "unknown_npz_key",
    "missing_npz_key",
    "invalid_recipe_json",
    "npz_read_failed",
]


class WaveformErrorData(TypedDict, total=False):
    """Optional diagnostics for the failing format constraint.

    name identifies idata/qdata/time; ndim is its observed dimensionality.
    *_shape are observed array shapes. sample_count is the observed count,
    max_sample_count the upper limit. time0 is the first time in microseconds;
    peak_abs is the observed I/Q magnitude. path identifies recipe_json.
    unknown_keys/missing_keys list the offending NPZ key names.
    Only fields relevant to the reason are present.
    """

    name: str
    ndim: int
    idata_shape: tuple[int, ...]
    qdata_shape: tuple[int, ...]
    time_shape: tuple[int, ...]
    sample_count: int
    max_sample_count: int
    time0: float
    peak_abs: float
    path: str
    unknown_keys: list[str]
    missing_keys: list[str]


class WaveformFormatError(ValueError):
    """Malformed sample arrays, NPZ bytes, or recipe JSON.

    reason is a stable format classification. data contains only relevant
    typed diagnostics, not paths to assets or repository state. The decoder
    retains an archive/JSON cause when translating its failure.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: WaveformErrorReason,
        data: WaveformErrorData | None = None,
    ) -> None:
        """Describe a format failure without interpreting repository policy."""
        super().__init__(message)
        self.reason = reason
        self.data: WaveformErrorData = {} if data is None else data


@dataclass(frozen=True)
class WaveformSamples:
    """One isolated read of normalized finite one-dimensional float64 arrays.

    time is the reference axis in microseconds, starts at zero, strictly
    increases, and contains 2..1,000,001 samples. idata is normalized real I;
    qdata is real Q of the same length or None when absent/all zero.
    The magnitude hypot(I,Q) is at most one. Each validated read owns its
    arrays, but callers can modify them; asset bytes are never frozen.
    """

    time: NDArray[np.float64]
    idata: NDArray[np.float64]
    qdata: NDArray[np.float64] | None


@dataclass(frozen=True)
class WaveformFile:
    """Decoded samples and optional opaque recipe text.

    samples obeys WaveformSamples constraints. recipe_json is a scalar JSON
    string or None when no recipe was supplied. JSON syntax is validated,
    but segments, normalization, and formula semantics are not interpreted.
    """

    samples: WaveformSamples
    recipe_json: str | None = None


def validate_samples(*, time: object, idata: object, qdata: object) -> WaveformSamples:
    """Validate and isolate array-like time/I/Q using WaveformSamples rules.

    qdata=None means zero Q. Return None for absent or entirely zero Q.
    Raise WaveformFormatError for nonnumeric, nonfinite, dimensionality,
    length, reference time, sample count, or magnitude violations.
    The returned arrays do not share caller-owned storage.
    """
    raise NotImplementedError("Shared sample validation is not connected")


def decode_npz(payload: bytes) -> WaveformFile:
    """Decode isolated samples from NPZ bytes with allow_pickle=False.

    Require idata/qdata/time; accept only optional recipe_json in addition.
    Validate arrays and scalar JSON syntax, not formula semantics. Reject
    unsupported/missing keys and malformed content with WaveformFormatError.
    No filesystem or repository access occurs.
    """
    raise NotImplementedError("Shared NPZ decoding is not connected")


def encode_npz(data: WaveformFile) -> bytes:
    """Validate samples/opaque JSON and encode idata/qdata/time to NPZ bytes.

    Always write qdata, including zero Q. Write recipe_json only when present.
    Preserve recipe text without interpreting it. Raise WaveformFormatError
    for invalid samples or JSON; do not write files or alter the input.
    """
    raise NotImplementedError("Shared NPZ encoding is not connected")
