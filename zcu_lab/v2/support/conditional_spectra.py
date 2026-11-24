"""Numerical mechanics shared by the two spectroscopy entries."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance

from zcu_lab.v2.support.paired_counts import (
    qubit_marginal_axis,
    qubit_marginal_probabilities,
)


@dataclass(frozen=True)
class SpectroscopyAnalysis:
    """Count-derived spectra on (DC, prepared_qubit1, prepared_qubit2, frequency).

    excitation adds a final Qubit1/Qubit2 axis. acquired_shots is the per-point sum
    after any herald selection. complete marks DC/preparation rows with at
    least one completed round at every frequency. Zero totals give NaN
    excitation. Other stays in the denominator; probabilities are not
    corrected for state-preparation or readout error.
    """

    excitation: NDArray[np.float64]
    acquired_shots: NDArray[np.float64]
    complete: NDArray[np.bool_]


def excitation_sem(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return terminal Qubit1/Qubit2 P(e) SEM; counts end in 3 × 3 (g/e/other).

    Other contributes to N. Fewer than two shots and unmeasured points give NaN.
    """
    errors = []
    for target in ("qubit1", "qubit2"):
        marginal = counts.sum(axis=qubit_marginal_axis(target))
        errors.append(np.sqrt(class_probability_covariance(marginal)[..., 1, 1]))
    return np.stack(errors, axis=-1)


def summarize_spectra(counts: NDArray[np.float64]) -> SpectroscopyAnalysis:
    """Compute Qubit1/Qubit2 excitation, accepted totals and all-frequency coverage.

    counts ends in terminal (3,3), with g/e/other on each axis and frequency
    immediately before them. Counts accumulate over completed rounds. Whole
    missing points may be NaN. Invalid counts or partly missing points raise
    ValueError. Zero totals yield NaN excitation.
    """
    if counts.ndim < 3 or counts.shape[-2:] != (3, 3):
        raise ValueError("Expected a frequency axis and terminal class axes (3,3)")
    finite = np.all(np.isfinite(counts) & (counts >= 0), axis=(-2, -1))
    missing = np.all(np.isnan(counts), axis=(-2, -1))
    if not np.all(finite | missing):
        raise ValueError("Spectroscopy contains invalid or partially missing counts")
    acquired_shots = counts.sum(axis=(-2, -1))
    if not np.all(np.isfinite(acquired_shots) | missing):
        raise ValueError("Acquired shots must be nonnegative totals or NaN")
    excitation = np.stack(
        [
            qubit_marginal_probabilities(counts, target)[..., 1]
            for target in ("qubit1", "qubit2")
        ],
        axis=-1,
    )
    return SpectroscopyAnalysis(
        excitation,
        acquired_shots,
        np.all(finite, axis=-1),
    )
