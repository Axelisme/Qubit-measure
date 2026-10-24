"""Interior readout-peak acceptance for wavelet-smoothed SNR landscapes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReadoutOptimum:
    """Unique interior optimum: axis indexes and native MHz/gain coordinates."""

    freq_index: int
    gain_index: int
    freq: float
    gain: float


def accepted_readout_optimum(
    landscape: NDArray[np.float64],
    freqs: NDArray[np.float64],
    gains: NDArray[np.float64],
) -> ReadoutOptimum | None:
    """Return a trustworthy interior readout optimum, or None for fallback.

    The SNR map is already smoothed before it reaches this helper. A feedback point
    is accepted only when the smoothed landscape has finite structure, a unique
    interior maximum, and that maximum rises above its local neighbourhood.
    """
    min_landscape_span = 1e-12
    min_local_prominence_fraction = 1e-3
    snr = np.asarray(landscape, dtype=np.float64)
    freq_axis = np.asarray(freqs, dtype=np.float64)
    gain_axis = np.asarray(gains, dtype=np.float64)
    if snr.shape != (freq_axis.size, gain_axis.size):
        raise ValueError(
            "ro_optimize landscape shape does not match axes: "
            f"{snr.shape} vs ({freq_axis.size}, {gain_axis.size})"
        )
    if (
        not np.all(np.isfinite(freq_axis))
        or not np.all(np.isfinite(gain_axis))
        or freq_axis.size < 3
        or gain_axis.size < 3
    ):
        return None

    finite = np.isfinite(snr)
    if not np.any(finite):
        return None
    finite_values = snr[finite]
    span = float(np.max(finite_values) - np.min(finite_values))
    masked = np.where(finite, snr, -np.inf)
    best_value = float(np.max(masked))
    best_locations = np.argwhere(masked == best_value)
    if (
        not np.isfinite(span)
        or span <= min_landscape_span
        or best_locations.shape[0] != 1
    ):
        return None
    best_fi = int(best_locations[0, 0])
    best_gi = int(best_locations[0, 1])
    if best_fi in (0, freq_axis.size - 1) or best_gi in (0, gain_axis.size - 1):
        return None

    local = masked[best_fi - 1 : best_fi + 2, best_gi - 1 : best_gi + 2].copy()
    local[1, 1] = -np.inf
    local_neighbors = local[np.isfinite(local)]
    local_prominence = best_value - float(np.max(local_neighbors, initial=-np.inf))
    min_prominence = max(
        min_landscape_span,
        min_local_prominence_fraction * span,
    )
    if local_neighbors.size == 0 or local_prominence < min_prominence:
        return None

    return ReadoutOptimum(
        freq_index=best_fi,
        gain_index=best_gi,
        freq=float(freq_axis[best_fi]),
        gain=float(gain_axis[best_gi]),
    )
