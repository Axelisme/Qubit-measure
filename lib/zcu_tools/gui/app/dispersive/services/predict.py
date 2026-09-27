"""PredictService — dispersive GUI adapter over the Fluxonium prediction engine.

The live g/r_f tuning recomputes the dispersive shift on every slider move. Cache
identity, fast/scqubits fallback, and fallback provenance live in
``zcu_tools.simulate.fluxonium``; this service binds the GUI's fixed resolution
and keeps the existing tuple-of-lines adapter API for the controller/UI.

The prediction always runs over the FULL preprocessed flux axis (no down-sampling):
the preprocessing already coarsens the spectrum, and the numpy fast path is quick
enough to predict every flux point on each accept. There is no ``step`` knob.

Pure, Qt-free, no State write. Used synchronously by the tuning canvas (no event).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.dispersive.predict import PREDICTION_RESOLUTION
from zcu_tools.simulate.fluxonium import FluxoniumPrediction, FluxoniumPredictionSession

logger = logging.getLogger(__name__)


class PredictService:
    """LRU-cached dispersive prediction for one (params, flux-axis) combination."""

    def __init__(
        self,
        params: tuple[float, float, float],
        sp_fluxs: NDArray[np.float64],
    ) -> None:
        engine = FluxoniumPrediction(params, resolution=PREDICTION_RESOLUTION)
        self._session: FluxoniumPredictionSession = engine.bind_flux_axis(sp_fluxs)

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
        result = self._session.predict_dispersive(g, bare_rf, return_dim=return_dim)
        if result.used_fallback:
            logger.warning(
                "fast dispersive labeling ambiguous (g=%s); using scqubits", g
            )
        return result.lines

    def flux_axis(self) -> NDArray[np.float64]:
        """The flux axis a ``predict`` call ran over (the full preprocessed axis)."""
        return self._session.flux_axis()
