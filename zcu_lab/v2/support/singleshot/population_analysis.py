"""Readout estimates and shot error bars for the circle one-dimensional analyses."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import corrected_population_sem
from zcu_tools.analysis.singleshot import correct_populations
from zcu_tools.plotting.plots import Plots

logger = logging.getLogger(__name__)

CLASSIFIED_FIGURE = "classified-fractions"
UNAVAILABLE_NOTICE = "Readout correction unavailable"


@dataclass(frozen=True)
class ReadoutSeries:
    values: NDArray[np.float64]
    sem: NDArray[np.float64]
    correction_unavailable: bool


def circle_readout_series(
    classified: NDArray[np.float64],
    counts: NDArray[np.float64],
    matrix: NDArray[np.float64] | None,
) -> ReadoutSeries:
    """Keep unclipped estimates, or diagnose a valid saved matrix with no solution.

    Missing matrices use classified fractions as before. If the solve fails,
    return classified fractions and their uncorrected shot SEM for diagnosis only.
    Malformed inputs keep their ordinary errors; missing classified points do not
    count as nonfinite solutions.
    """
    corrected = _solve(classified, matrix)
    if corrected is None:
        return ReadoutSeries(classified, corrected_population_sem(counts, None), True)
    return ReadoutSeries(corrected, corrected_population_sem(counts, matrix), False)


def _solve(
    classified: NDArray[np.float64], matrix: NDArray[np.float64] | None
) -> NDArray[np.float64] | None:
    if matrix is None:
        return classified
    classes = classified.shape[-1]
    if matrix.shape != (classes, classes) or not np.isfinite(matrix).all():
        # Only a valid saved matrix qualifies for a figures-only diagnostic.
        return correct_populations(classified, matrix, clip=False)

    try:
        corrected = correct_populations(classified, matrix, clip=False)
    except np.linalg.LinAlgError as error:
        logger.warning("%s: singular confusion matrix (%s)", UNAVAILABLE_NOTICE, error)
        return None

    failed = np.isfinite(classified).all(axis=-1) & ~np.isfinite(corrected).all(axis=-1)
    if failed.any():
        logger.warning(
            "%s: nonfinite corrected values for %d finite classified points",
            UNAVAILABLE_NOTICE,
            np.count_nonzero(failed),
        )
        return None
    return corrected


def plot_t1_classified_fractions(
    plots: Plots,
    delays: NDArray[np.float64],
    series: ReadoutSeries,
    panels: tuple[tuple[str, str, str], ...],
    *,
    title: str,
    xlabel: str,
) -> None:
    """Publish both preparations' observed fractions without a rate model."""
    fig, _ = plots.subplots(
        num=CLASSIFIED_FIGURE, nrows=2, ncols=3, figsize=(10, 6), layout="constrained"
    )
    fig.suptitle(f"{title} · qubit · classified fractions\n{UNAVAILABLE_NOTICE}")
    for index, (ax, (branch, state, color)) in enumerate(
        zip(fig.axes, panels, strict=True)
    ):
        prep, outcome = divmod(index, 3)
        ax.errorbar(
            delays,
            series.values[:, prep, outcome],
            yerr=series.sem[:, prep, outcome],
            fmt="o",
            markersize=3,
            capsize=2,
            color=color,
            label="Data ± shot SEM",
        )
        ax.set(xlabel=xlabel, ylabel=f"Fraction ({state})", title=branch)
    fig.axes[2].legend(loc="best", fontsize=10)
