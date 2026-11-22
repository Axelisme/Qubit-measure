"""Plot data for wrapped phases without lines across the branch cut."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def wrapped_phase_plot_data(
    x: NDArray[np.int64] | NDArray[np.float64],
    phase_deg: Sequence[float] | NDArray[np.float64],
    *aligned: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Return x, wrapped degrees and any aligned vectors (e.g. fit SE) for plotting.

    Supply matching 1D vectors in scan order and use NaN for a missing angle.
    Add NaN separators for adjacent plotted angles at least 180° apart.
    Matplotlib breaks the line there while keeping each original marker and error bar.
    """
    phases = np.asarray(phase_deg, dtype=np.float64)
    breaks = np.flatnonzero(np.abs(np.diff(phases)) >= 180) + 1
    return tuple(
        np.insert(values, breaks, np.nan)
        for values in (np.asarray(x, dtype=np.float64), phases, *aligned)
    )
