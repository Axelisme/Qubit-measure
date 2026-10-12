"""Complete synthetic numeric results for application lifecycle tests."""

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.search import (
    DatabaseSearchResult,
    ParamBounds,
    SearchQuality,
    TransitionResidual,
)


def perfect_fit_result(
    fluxs: NDArray[np.float64], freqs: NDArray[np.float64], bounds: ParamBounds,
) -> DatabaseSearchResult:
    """Return one (5, 1, 0.5) candidate matching paired flux/GHz points exactly.

    All points belong to transitions 0-1, with zero residual and no outliers.
    Arrays are detached from the caller. bounds retains the captured input.
    """
    return DatabaseSearchResult(
        params=(5.0, 1.0, 0.5), best_distance=0.0, best_scale=1.0, best_index=0,
        entry_results=np.array([[0.0, 1.0]]), entry_params=np.array([[5.0, 1.0, 0.5]]),
        fluxs=fluxs.copy(), freqs=freqs.copy(), predicted_freqs=freqs.copy(), bounds=bounds,
        transition_indices=np.zeros(freqs.size, dtype=np.intp),
        residuals=np.zeros(freqs.size), outlier_mask=np.zeros(freqs.size, dtype=np.bool_),
        quality=SearchQuality(mean=0.0, outliers=0, outlier_threshold=0.0,
            per_transition=(TransitionResidual("transitions 0-1", freqs.size, 0.0, 0.0),)),
    )
