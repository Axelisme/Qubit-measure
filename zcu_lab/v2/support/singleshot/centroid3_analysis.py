"""Analyze saved g/e/f counts without touching acquisition or classification."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import class_probability_covariance
from zcu_tools.analysis.singleshot import Centroid3Summary, correct_populations
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg

from .util import centroid3_confusion_matrix


def correct_saved_centroid3(
    readout: PulseReadoutCfg, classified: NDArray[np.float64]
) -> tuple[NDArray[np.float64] | None, Centroid3Summary]:
    """Return optional unclipped correction and diagnostics for the applied matrix.

    The caller validates and normalizes its counts, keeping the scan's original
    completeness policy. A missing/unusable matrix leaves all points classified.
    """
    if classified.ndim == 0 or classified.shape[-1] != 3:
        raise ValueError("Centroid3 populations require a final g/e/f axis")
    if not np.isfinite(classified).all():
        raise ValueError("Centroid3 populations must be finite")

    matrix = centroid3_confusion_matrix(readout, ro_ch=readout.ro_ch)
    corrected = correct_populations(
        classified, matrix, clip=False, fallback_on_nonfinite=True
    )
    if corrected is classified:
        return None, Centroid3Summary.from_applied_matrix(None)
    return corrected, Centroid3Summary.from_applied_matrix(matrix)


def centroid3_trace_sems(
    counts: NDArray[np.float64], confusion: Centroid3Summary
) -> tuple[NDArray[np.float64], ...]:
    """Return classified shot SEM and, if applied, corrected shot SEM."""
    measured = class_probability_covariance(counts)
    covariances = (measured,)
    if confusion.confusion_matrix is not None:
        correction = np.linalg.inv(confusion.confusion_matrix.T)
        with np.errstate(over="ignore", invalid="ignore"):
            covariances += (correction @ measured @ correction.T,)
    return tuple(
        np.sqrt(np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 0.0))
        for cov in covariances
    )


def centroid3_title(scan: str, confusion: Centroid3Summary) -> str:
    """Name the population basis and report matrix quality only if corrected."""
    if confusion.confusion_matrix is None:
        return f"{scan} · g → f · classified populations"
    condition = confusion.condition_number
    minimum = confusion.minimum_singular_value
    condition_text = "unavailable" if condition is None else f"{condition:.3g}"
    minimum_text = "unavailable" if minimum is None else f"{minimum:.3g}"
    return (
        f"{scan} · g → f · preparation-label fractions\n"
        f"Readout condition number: {condition_text} · Minimum singular value: {minimum_text}"
    )
