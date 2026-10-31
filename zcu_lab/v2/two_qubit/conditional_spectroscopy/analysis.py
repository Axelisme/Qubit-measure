"""Numerical summaries of classified spectroscopy counts, without plotting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._sums import marginal_probabilities


@dataclass(frozen=True)
class SpectroscopyAnalysis:
    """Count-derived spectra on (DC, prepared_f1, prepared_f2, frequency).

    excitation adds a final F1/F2 axis. acquired_shots is the per-point total,
    before herald selection when applicable. complete omits frequency and marks
    preparations with every requested reps × rounds. Partial counts enter P_e;
    zero counts give NaN. Other remains in the denominator. These probabilities
    are not confusion-corrected energy-state populations.
    """

    excitation: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    complete: NDArray[np.bool_]


def summarize_spectra(
    counts: NDArray[np.float64],
    acquired_shots: NDArray[np.float64],
    requested_shots: int,
) -> SpectroscopyAnalysis:
    """Compute F1/F2 P_e and frequency completeness from terminal counts.

    counts ends in terminal (3,3), with g/e/other on each axis. acquired_shots
    matches the preceding scan axes and counts all shots before selection.
    Whole missing points may be NaN. Invalid or partly missing counts raise
    ValueError. Zero totals give NaN without warnings; partial counts are used.
    """
    if counts.shape[-2:] != (3, 3) or counts.shape[:-2] != acquired_shots.shape:
        raise ValueError("Terminal counts and acquired shot coordinates must match")
    finite = np.all(np.isfinite(counts) & (counts >= 0), axis=(-2, -1))
    missing = np.all(np.isnan(counts), axis=(-2, -1))
    if not np.all(finite | missing):
        raise ValueError("Spectroscopy contains invalid or partially missing counts")
    if not np.all(
        np.isnan(acquired_shots) | (np.isfinite(acquired_shots) & (acquired_shots >= 0))
    ):
        raise ValueError("Acquired shots must be nonnegative totals or NaN")
    excitation = np.stack(
        [marginal_probabilities(counts, target)[..., 1] for target in ("f1", "f2")],
        axis=-1,
    )
    return SpectroscopyAnalysis(
        excitation,
        acquired_shots,
        np.all(acquired_shots >= requested_shots, axis=-1),
    )
