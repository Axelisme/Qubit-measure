"""PredictService — the LRU-cached dispersive simulation (notebook ``get_dispersive``).

The live g/r_f tuning recomputes the dispersive shift on every slider move; the
scqubits ``calculate_dispersive_vs_flux`` is expensive, so this caches results
keyed on the full parameter tuple. The cache is bound to one (fluxonium params,
flux axis) combination — rebuild the service when the one-tone or fit inputs
change so a stale axis cannot be served.

Pure, Qt-free, no State write. Used synchronously by the tuning canvas (no event).
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium import (
    DressedLabelingError,
    calculate_dispersive_vs_flux,
    calculate_dispersive_vs_flux_fast,
)

logger = logging.getLogger(__name__)

# Fluxonium Hilbert-space resolution, fixed (the notebook's defaults). Not
# user-tunable in the GUI — only ``step`` (flux down-sampling) is.
_QUB_DIM = 15
_QUB_CUTOFF = 30
_RES_DIM = 4


class PredictService:
    """LRU-cached dispersive prediction for one (params, flux-axis) combination."""

    def __init__(
        self,
        params: tuple[float, float, float],
        sp_fluxs: NDArray[np.float64],
    ) -> None:
        self._params = params
        self._sp_fluxs = np.asarray(sp_fluxs, dtype=np.float64)

        @lru_cache(maxsize=None)
        def _cached(
            g: float,
            bare_rf: float,
            step: int,
            return_dim: int,
        ) -> tuple[NDArray[np.float64], ...]:
            fluxs = self._sp_fluxs[::step]
            # The numpy-only fast path is ~9x faster and matches scqubits to
            # 0.00000 MHz for the dispersive (low-level) regime. If its simple
            # dressed-state labeling is ever ambiguous (strong coupling / dense
            # levels), fall back to the robust scqubits ParameterSweep.
            try:
                return calculate_dispersive_vs_flux_fast(
                    self._params,
                    fluxs,
                    bare_rf,
                    g,
                    res_dim=_RES_DIM,
                    qub_cutoff=_QUB_CUTOFF,
                    qub_dim=_QUB_DIM,
                    return_dim=return_dim,
                )
            except DressedLabelingError:
                logger.warning(
                    "fast dispersive labeling ambiguous (g=%s); using scqubits", g
                )
                return calculate_dispersive_vs_flux(
                    self._params,
                    fluxs,
                    bare_rf,
                    g,
                    progress=False,
                    res_dim=_RES_DIM,
                    qub_cutoff=_QUB_CUTOFF,
                    qub_dim=_QUB_DIM,
                    return_dim=return_dim,
                )

        self._cached = _cached

    def predict(
        self,
        g: float,
        bare_rf: float,
        *,
        step: int = 1,
        return_dim: int = 2,
    ) -> tuple[NDArray[np.float64], ...]:
        """The dispersive-shifted resonator frequencies (GHz) for these inputs.

        Returns ``return_dim`` arrays (rf_0, rf_1, ...) each over the down-sampled
        flux axis ``sp_fluxs[::step]`` (qub_dim / qub_cutoff / res_dim fixed).
        Cache-keyed on every argument, so a repeated slider position is free.
        """
        return self._cached(g, bare_rf, step, return_dim)

    def flux_axis(self, step: int) -> NDArray[np.float64]:
        """The down-sampled flux axis matching a ``predict(..., step=step)`` call."""
        return self._sp_fluxs[::step]
