"""Shared numeric mechanics for simultaneous qubit spectroscopy and Amp Rabi."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MuxSweepResult:
    """Zipped average IQ with independent hardware-quantized coordinates.

    tone_indices is int64 (K,), in original full-readout index order selected by
    the caller, not ADC order. coordinates is float64 (K,P), in MHz for frequency
    scans or normalized gain for Amp Rabi. signals is complex128 (K,P), averaged
    I+jQ in the program's ADC-length-normalized a.u. Unwritten points are NaN;
    completed columns survive cancellation. Check the RunContext stop signal to
    distinguish acquisition errors from stopped or complete results.
    """

    tone_indices: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    signals: NDArray[np.complex128]


def validate_mux_result(result: MuxSweepResult) -> None:
    """Reject inconsistent array shapes, indices and nonfinite coordinates.

    Accept NaN IQ for partial runs. Require unique nonnegative original int64
    indices (K,), K>0, coordinates and signals (K,P), P>0. No data is changed.
    Raises ValueError; this is a source-contract check, not a numerical fit.
    """
    indices = result.tone_indices
    if (
        indices.ndim != 1
        or indices.dtype != np.dtype(np.int64)
        or not indices.size
        or np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("tone_indices must be nonempty unique nonnegative integers")
    if (
        result.coordinates.ndim != 2
        or result.coordinates.dtype != np.dtype(np.float64)
        or result.signals.dtype != np.dtype(np.complex128)
        or result.coordinates.shape[0] != len(indices)
        or result.coordinates.shape[1] == 0
        or result.signals.shape != result.coordinates.shape
        or not np.all(np.isfinite(result.coordinates))
    ):
        raise ValueError(
            "coordinates float64 and signals complex128 must be (K,P), with finite coordinates"
        )


@dataclass(frozen=True)
class MuxFitFailure:
    """Numerical failure for original tone index, with the fitter's reason."""

    index: int
    reason: str
