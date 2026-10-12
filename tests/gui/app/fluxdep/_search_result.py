"""Complete synthetic numeric results for application lifecycle tests."""

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.search import (
    DatabaseSearchResult,
    ParamBounds,
    SearchQuality,
    TransitionResidual,
)
from zcu_tools.gui.app.fluxdep.services.fit import SearchInput
from zcu_tools.gui.app.fluxdep.state import FluxDepState, transitions_with_freqs


def synthetic_search_input(
    state: FluxDepState,
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    bounds: ParamBounds,
) -> SearchInput:
    """Detached capture of the published fit choices over an explicit cloud.

    Mirrors FitService.capture_search without requiring selected points, so a
    test can publish a synthetic result on an empty collection. fluxs/freqs
    are copied normalized-flux/GHz coordinates; stages snapshot the current
    dependency graph on the owner thread.
    """
    fit = state.fit
    return SearchInput(
        search_db=fit.search_db,
        fluxs=fluxs.copy(),
        freqs=freqs.copy(),
        transitions=deepcopy(
            transitions_with_freqs(fit.transitions, fit.r_f, fit.sample_f)
        ),
        bounds=bounds,
        stages=state.pipeline_snapshot(),
    )


def perfect_fit_result(
    fluxs: NDArray[np.float64],
    freqs: NDArray[np.float64],
    bounds: ParamBounds,
) -> DatabaseSearchResult:
    """Return one (5, 1, 0.5) candidate matching paired flux/GHz points exactly.

    All points belong to transitions 0-1, with zero residual and no outliers.
    Arrays are detached from the caller. bounds retains the captured input.
    """
    return DatabaseSearchResult(
        params=(5.0, 1.0, 0.5),
        best_distance=0.0,
        best_scale=1.0,
        best_index=0,
        entry_results=np.array([[0.0, 1.0]]),
        entry_params=np.array([[5.0, 1.0, 0.5]]),
        fluxs=fluxs.copy(),
        freqs=freqs.copy(),
        predicted_freqs=freqs.copy(),
        bounds=bounds,
        transition_indices=np.zeros(freqs.size, dtype=np.intp),
        residuals=np.zeros(freqs.size),
        outlier_mask=np.zeros(freqs.size, dtype=np.bool_),
        quality=SearchQuality(
            mean=0.0,
            outliers=0,
            outlier_threshold=0.0,
            per_transition=(
                TransitionResidual("transitions 0-1", freqs.size, 0.0, 0.0),
            ),
        ),
    )
