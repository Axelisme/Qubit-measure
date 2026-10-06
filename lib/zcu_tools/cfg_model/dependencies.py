"""Typed dependency contracts shared by schema and asset owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class WaveformSamples:
    """One isolated asset read, with finite one-dimensional float64 arrays.

    time is the reference axis in microseconds, starts at zero and strictly
    increases. idata is normalized real I data of the same length. qdata is
    normalized real Q data of the same length, or None if all Q values are zero.
    Arrays belong to this read; changing them cannot change the stored asset or
    arrays returned by another read. Providers validate archive data, not callers.
    """

    time: NDArray[np.float64]
    idata: NDArray[np.float64]
    qdata: NDArray[np.float64] | None


class WaveformAssetInfo(Protocol):
    """Asset reads bound to one explicit scope, without caching asset bytes.

    Keys identify assets in the bound scope, not paths or GUI roles. Each call
    reads current bytes. Missing or invalid assets and I/O failures raise an
    owner-defined exception; no empty data or duration fallback is permitted.
    Copying a consumer must not rebind this dependency to another scope.
    """

    def duration(self, data_key: str) -> float:
        """Return current asset time[-1] in microseconds; propagate read errors."""
        ...

    def samples(self, data_key: str) -> WaveformSamples:
        """Read isolated, validated I/Q samples and their reference time axis."""
        ...
