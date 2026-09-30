"""Dispersive single-point prediction at the application resolution."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium import FluxoniumPrediction, PredictionResolution

logger = logging.getLogger(__name__)

# Fluxonium Hilbert-space resolution, fixed (the notebook's defaults). Not
# user-tunable in the GUI.
_QUB_DIM = 15
_QUB_CUTOFF = 30
_RES_DIM = 4
PREDICTION_RESOLUTION = PredictionResolution(
    qub_dim=_QUB_DIM,
    qub_cutoff=_QUB_CUTOFF,
    res_dim=_RES_DIM,
)


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
    a handful of *arbitrary* fluxs (not the preprocessed axis), so it uses the
    engine's stateless prediction path rather than the axis-bound session cache.
    Used synchronously on the Qt main thread (cheap enough for drag feedback); no
    State write, no event.
    """
    engine = FluxoniumPrediction(params, resolution=PREDICTION_RESOLUTION)
    result = engine.predict_dispersive(fluxs, g, bare_rf, return_dim=return_dim)
    if result.used_fallback:
        logger.warning("fast sample-point labeling ambiguous (g=%s); using scqubits", g)
    return result.lines
