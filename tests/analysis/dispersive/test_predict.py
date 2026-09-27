"""Single-point prediction contracts at arbitrary flux coordinates."""

from __future__ import annotations

import numpy as np
import zcu_tools.simulate.fluxonium.prediction as prediction_mod
from zcu_tools.analysis.dispersive.predict import predict_dispersive_at


def test_predict_dispersive_at_arbitrary_fluxs(monkeypatch):
    # the live single-point path for sample lines: arbitrary fluxs, calls the fast
    # path directly (no axis binding), returns return_dim arrays.
    seen = {}

    def fake_calc(
        params, fluxs, bare_rf, g, *, res_dim, qub_cutoff, qub_dim, return_dim
    ):
        seen["fluxs"] = np.asarray(fluxs)
        return tuple(np.full(len(fluxs), g + 0.1 * k) for k in range(return_dim))

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", fake_calc)

    fluxs = np.array([0.12, 0.31, 0.47])
    rf_0, rf_1 = predict_dispersive_at((4.0, 1.0, 0.5), fluxs, 0.06, 5.3)
    assert len(rf_0) == 3 and len(rf_1) == 3
    np.testing.assert_allclose(seen["fluxs"], fluxs)  # passed through verbatim


def test_predict_dispersive_at_falls_back_on_ambiguous_labeling(monkeypatch):
    # a DressedLabelingError from the fast path falls back to scqubits.
    from zcu_tools.simulate.fluxonium import DressedLabelingError

    def boom(*a, **k):
        raise DressedLabelingError("ambiguous")

    calls = []

    def fallback(
        params, fluxs, bare_rf, g, *, progress, res_dim, qub_cutoff, qub_dim, return_dim
    ):
        calls.append("scqubits")
        return tuple(np.zeros(len(fluxs)) for _ in range(return_dim))

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", boom)
    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux", fallback)

    predict_dispersive_at((4.0, 1.0, 0.5), np.array([0.3]), 0.5, 5.3)
    assert calls == ["scqubits"]
