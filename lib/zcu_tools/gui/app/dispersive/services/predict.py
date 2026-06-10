"""PredictService — the LRU-cached dispersive simulation (notebook ``get_dispersive``).

The live g/r_f tuning recomputes the dispersive shift on every slider move; the
scqubits ``calculate_dispersive_vs_flux`` is expensive, so this caches results
keyed on the full parameter tuple. The cache is bound to one (fluxonium params,
flux axis) combination — rebuild the service when the one-tone or fit inputs
change so a stale axis cannot be served.

The prediction always runs over the FULL preprocessed flux axis (no down-sampling):
the preprocessing already coarsens the spectrum, and the numpy fast path is quick
enough to predict every flux point on each accept. There is no ``step`` knob.

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
# user-tunable in the GUI.
_QUB_DIM = 15
_QUB_CUTOFF = 30
_RES_DIM = 4


def predict_dispersive_at(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    g: float,
    bare_rf: float,
    *,
    return_dim: int = 2,
) -> tuple[NDArray[np.float64], ...]:
    """Dispersive ground/excited resonator frequencies (GHz) at arbitrary fluxs.

    The live single-point path for the draggable sample-flux lines: it predicts at
    a handful of *arbitrary* fluxs (not the preprocessed axis), so it can't use the
    axis-bound LRU ``PredictService`` — it calls the numpy fast path directly (one
    eigensolve per flux, a few ms each) with the same scqubits fallback on an
    ambiguous dressed labeling. Used synchronously on the Qt main thread (cheap
    enough for drag feedback); no State write, no event.
    """
    try:
        return calculate_dispersive_vs_flux_fast(
            params,
            fluxs,
            bare_rf,
            g,
            res_dim=_RES_DIM,
            qub_cutoff=_QUB_CUTOFF,
            qub_dim=_QUB_DIM,
            return_dim=return_dim,
        )
    except DressedLabelingError:
        logger.warning("fast sample-point labeling ambiguous (g=%s); using scqubits", g)
        return calculate_dispersive_vs_flux(
            params,
            fluxs,
            bare_rf,
            g,
            progress=False,
            res_dim=_RES_DIM,
            qub_cutoff=_QUB_CUTOFF,
            qub_dim=_QUB_DIM,
            return_dim=return_dim,
        )


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
            return_dim: int,
        ) -> tuple[NDArray[np.float64], ...]:
            fluxs = self._sp_fluxs
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
        return_dim: int = 2,
    ) -> tuple[NDArray[np.float64], ...]:
        """The dispersive-shifted resonator frequencies (GHz) for these inputs.

        Returns ``return_dim`` arrays (rf_0, rf_1, ...) each over the full
        preprocessed flux axis (qub_dim / qub_cutoff / res_dim fixed). Cache-keyed
        on every argument, so a repeated slider position is free.
        """
        return self._cached(g, bare_rf, return_dim)

    def flux_axis(self) -> NDArray[np.float64]:
        """The flux axis a ``predict`` call ran over (the full preprocessed axis)."""
        return self._sp_fluxs
