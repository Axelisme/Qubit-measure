"""Readout estimates and shot error bars for the circle one-dimensional analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import corrected_population_sem
from zcu_tools.analysis.singleshot import try_correct_populations

CLASSIFIED_FIGURE = "classified-fractions"
UNAVAILABLE_NOTICE = "Readout correction unavailable"


@dataclass(frozen=True)
class ReadoutSeries:
    """Plot data with the same shape and class order as the input fractions.

    ``values`` contains unclipped corrected estimates, or raw classified fractions
    when no matrix is supplied or correction is unavailable. ``sem`` contains the
    matching shot SEM in fraction units, with NaN for an unavailable bar.
    ``correction_unavailable`` marks diagnostic-only data that must not feed a fit
    or summary. It is false for a missing matrix or an unavailable SEM alone.
    """

    values: NDArray[np.float64]
    sem: NDArray[np.float64]
    correction_unavailable: bool


def circle_readout_series(
    classified: NDArray[np.float64],
    counts: NDArray[np.float64],
    matrix: NDArray[np.float64] | None,
) -> ReadoutSeries:
    """Combine point estimates with shot SEM for a circle readout analysis.

    Supply aligned ``classified`` fractions and retained ``counts`` of shape
    ``(..., K)`` with K >= 2 and the same final-axis class order. Counts follow
    ``class_probability_covariance``: nonnegative integral values or NaN for
    missing counts. ``matrix`` is None or a finite (K, K) confusion matrix with
    true-state rows and classified-state columns, treated as fixed.

    None leaves the fractions uncorrected without marking a failure. A singular
    matrix or a nonfinite solution at a fully finite input point is logged and
    returns raw fractions and raw SEM for diagnosis. An already missing input
    point does not trigger this fallback. Invalid counts or a malformed matrix
    raise ValueError; only failures of a valid solve are diagnosed.
    """
    corrected = try_correct_populations(classified, matrix)
    if corrected is None:
        return ReadoutSeries(classified, corrected_population_sem(counts, None), True)
    return ReadoutSeries(corrected, corrected_population_sem(counts, matrix), False)
