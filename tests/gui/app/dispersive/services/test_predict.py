"""Tests for dispersive PredictService — LRU-cached dispersive prediction.

The default ``calculate_dispersive_vs_flux_fast`` (numpy, the predictor's path) is
monkeypatched to a cheap stub so the cache/passthrough wiring is tested without the
real eigensolve.
"""

from __future__ import annotations

import numpy as np
import zcu_tools.simulate.fluxonium.prediction as prediction_mod
from zcu_tools.gui.app.dispersive.services.predict import (
    PredictService,
)


def test_predict_covers_full_axis_and_caches(monkeypatch):
    calls = []

    def fake_calc(
        params, fluxs, bare_rf, g, *, res_dim, qub_cutoff, qub_dim, return_dim
    ):
        calls.append((g, bare_rf, res_dim, len(fluxs), return_dim))
        return tuple(np.full(len(fluxs), g + k) for k in range(return_dim))

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", fake_calc)

    sp_fluxs = np.linspace(0.0, 1.0, 20).astype(np.float64)
    svc = PredictService(params=(4.0, 1.0, 0.5), sp_fluxs=sp_fluxs)

    rf = svc.predict(0.06, 5.3, return_dim=2)
    assert len(rf) == 2
    assert len(rf[0]) == 20  # the full preprocessed axis (no down-sampling)
    assert calls[0] == (0.06, 5.3, 4, 20, 2)

    # same args → cache hit (no new call)
    svc.predict(0.06, 5.3, return_dim=2)
    assert len(calls) == 1

    # different g → new call
    svc.predict(0.07, 5.3, return_dim=2)
    assert len(calls) == 2


def test_flux_axis_is_full_preprocessed_axis():
    sp_fluxs = np.linspace(0.0, 1.0, 20).astype(np.float64)
    svc = PredictService(params=(4.0, 1.0, 0.5), sp_fluxs=sp_fluxs)
    np.testing.assert_allclose(svc.flux_axis(), sp_fluxs)
