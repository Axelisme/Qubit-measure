"""Analyze saved g/e/f counts without touching acquisition or classification."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.shot_uncertainty import corrected_population_sem
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
    classified = corrected_population_sem(counts, None)
    if confusion.confusion_matrix is None:
        return (classified,)
    return classified, corrected_population_sem(counts, confusion.confusion_matrix)
