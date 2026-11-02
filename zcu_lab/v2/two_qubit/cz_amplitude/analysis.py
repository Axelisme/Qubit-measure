"""Count-derived joint return probabilities and the shared first-return fit."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import FitQuality

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
class CZAmplitudeData:
    """Count-derived joint populations and zero-phase first-cycle fit."""

    data: CZReturnData
    fit: CZReturnFit


@dataclass(frozen=True)
class CZReturnSummary:
    """First-return scale and fixed-size zero-phase cosine diagnostics.

    scale is the measured first positive period, or None with reason. period
    retains a finite fit even when coverage prevents a proposal; error is its
    1σ uncertainty. offset and contrast are joint probabilities. fit_quality
    describes the actual fit without imposing an acceptance threshold.
    """

    scale: float | None
    period: float | None
    error: float | None
    offset: float | None
    contrast: float | None
    fit_quality: FitQuality | None
    reason: str | None


@dataclass(frozen=True)
class CZAmplitudeAnalysis:
    """Fixed-size first-return fit; measured probabilities remain plot-local."""

    fit: CZReturnSummary


def summarize_return(fit: CZReturnFit) -> CZAmplitudeAnalysis:
    """Retain fit values for writeback, not the curve or per-scale counts."""
    return CZAmplitudeAnalysis(
        CZReturnSummary(
            fit.scale,
            fit.period,
            fit.error,
            fit.offset,
            fit.contrast,
            fit.fit_quality,
            fit.reason,
        )
    )


def analyze_return(
    scales: NDArray[np.float64],
    counts: NDArray[np.float64],
    *,
    acquired_shots: NDArray[np.float64],
    expected_shots: int,
) -> CZAmplitudeData:
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
    return CZAmplitudeData(data, fit)
