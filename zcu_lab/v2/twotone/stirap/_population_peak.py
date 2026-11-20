"""Select a peak from STIRAP's confusion-corrected scan values."""

import numpy as np
from numpy.typing import NDArray


def first_finite_peak(populations: NDArray[np.float64]) -> int | None:
    """Return the flat index of the largest finite population, or None if absent.

    Pass one state's scan grid with the fastest acquisition axis last.
    Ties go to the first point in row-major acquisition order.
    """
    finite = np.isfinite(populations)
    if not finite.any():
        return None
    return int(np.argmax(np.where(finite, populations, -np.inf)))
