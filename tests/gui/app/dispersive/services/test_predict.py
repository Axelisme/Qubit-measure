"""Tests for the public flux-axis contract of PredictService."""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.dispersive.services.predict import (
    PredictService,
)


def test_flux_axis_is_full_preprocessed_axis():
    sp_fluxs = np.linspace(0.0, 1.0, 20).astype(np.float64)
    svc = PredictService(params=(4.0, 1.0, 0.5), sp_fluxs=sp_fluxs)
    np.testing.assert_allclose(svc.flux_axis(), sp_fluxs)
