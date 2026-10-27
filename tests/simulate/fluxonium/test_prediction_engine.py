from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux
from zcu_tools.simulate.fluxonium.prediction import (
    FluxoniumPrediction,
    PredictionResolution,
)


def test_prediction_resolution_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="qub_dim"):
        PredictionResolution(qub_dim=0)
    with pytest.raises(ValueError, match="qub_cutoff"):
        PredictionResolution(qub_cutoff=-1)
    with pytest.raises(ValueError, match="res_dim"):
        PredictionResolution(res_dim=0)


def test_predict_dispersive_uses_fast_backend_with_provenance() -> None:
    resolution = PredictionResolution(qub_dim=7, qub_cutoff=11, res_dim=3)
    engine = FluxoniumPrediction((4.0, 1.0, 0.5), resolution=resolution)
    fluxs = np.array([0.1, 0.2], dtype=np.float64)
    result = engine.predict_dispersive(fluxs, g=0.0, bare_rf=5.3, return_dim=2)

    assert result.backend == "fast"
    assert not result.used_fallback
    assert len(result.lines) == 2
    np.testing.assert_allclose(result.lines, np.full((2, 2), 5.3))


def test_axis_bound_session_caches_by_slider_args_and_owns_axis() -> None:
    axis = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    engine = FluxoniumPrediction((4.0, 1.0, 0.5))
    session = engine.bind_flux_axis(axis)
    axis[0] = 9.9

    first = session.predict_dispersive(g=0.06, bare_rf=5.3, return_dim=2)
    second = session.predict_dispersive(g=0.06, bare_rf=5.3, return_dim=2)
    third = session.predict_dispersive(g=0.07, bare_rf=5.3, return_dim=2)

    assert first is second
    assert third is not first
    np.testing.assert_allclose(session.flux_axis(), [0.1, 0.2, 0.3])
    assert not session.flux_axis().flags.writeable
    for g, result in [(0.06, first), (0.07, third)]:
        expected = engine.predict_dispersive(
            np.array([0.1, 0.2, 0.3]), g=g, bare_rf=5.3
        )
        np.testing.assert_allclose(result.lines, expected.lines)


def test_frequency_curves_use_reduced_flux_and_mhz_units() -> None:
    engine = FluxoniumPrediction((4.0, 1.0, 0.5))
    values = np.array([-0.2, 0.0, 0.3], dtype=np.float64)
    fluxs, freqs = engine.predict_frequencies_mhz(values, ((0, 1), (1, 3)))

    expected_fluxs = values
    _, energies = calculate_energy_vs_flux(
        (4.0, 1.0, 0.5), expected_fluxs, cutoff=40, evals_count=8
    )
    np.testing.assert_allclose(fluxs, expected_fluxs)
    np.testing.assert_allclose(freqs[0], (energies[:, 1] - energies[:, 0]) * 1e3)
    np.testing.assert_allclose(freqs[1], (energies[:, 3] - energies[:, 1]) * 1e3)
