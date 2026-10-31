"""Count-derived joint return probabilities and the shared first-return fit."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .._sums import joint_probabilities
from .fit import CZReturnFit, fit_return

OUTCOME_LABELS = tuple(
    f"{f1},{f2}" for f1 in ("g", "e", "other") for f2 in ("g", "e", "other")
)


@dataclass(frozen=True)
class CZReturnData:
    """counts/probabilities are (Ns,3,3), F1/F2 classes ordered g/e/other.

    complete marks scales with every requested shot acquired and nonzero accepted
    counts. acquired_shots includes all herald classes; analyzed_shots is the
    selected terminal total, including other. Zero totals give NaN probabilities.
    """

    counts: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    complete: NDArray[np.bool_]
    acquired_shots: NDArray[np.float64]
    analyzed_shots: NDArray[np.float64]


@dataclass(frozen=True)
class CZAmplitudeAnalysis:
    """Count-derived joint populations and zero-phase first-cycle fit."""

    data: CZReturnData
    fit: CZReturnFit


def analyze_return(
    scales: NDArray[np.float64],
    counts: NDArray[np.float64],
    *,
    acquired_shots: NDArray[np.float64],
    expected_shots: int,
) -> CZAmplitudeAnalysis:
    """Fit P00 from terminal counts, retaining other in the denominator.

    counts is (Ns,3,3), optionally already selected on herald (g,g).
    acquired_shots is the total before selection; expected_shots is reps*rounds.
    Fit only the contiguous fully acquired prefix with accepted shots, preserving
    first-cycle proposal policy. Partial scales remain visible but not fit.
    Wrong count/axis shapes raise ValueError. No cfg/library mutation occurs.
    """
    if counts.shape != (len(scales), 3, 3) or acquired_shots.shape != scales.shape:
        raise ValueError(
            "CZ return requires scales (Ns,), counts (Ns,3,3) and shot totals (Ns,)"
        )
    accepted = counts.sum(axis=(-2, -1))
    data = CZReturnData(
        counts=counts,
        probabilities=joint_probabilities(counts),
        complete=(acquired_shots == expected_shots) & (accepted > 0),
        acquired_shots=acquired_shots,
        analyzed_shots=accepted,
    )
    fit = fit_return(scales, data.probabilities[:, 0, 0], data.complete, accepted)
    return CZAmplitudeAnalysis(data, fit)

