from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.simulate.fluxonium import DressedLabelingError
from zcu_tools.simulate.fluxonium.prediction import (
    FluxAffineMap,
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


def test_flux_affine_roundtrip_and_array_mapping() -> None:
    affine = FluxAffineMap(flux_half=0.1, flux_period=0.8, flux_bias=0.03)
    values = np.array([-0.2, 0.0, 0.3], dtype=np.float64)

    expected = (values + 0.03 - 0.1) / 0.8 + 0.5
    np.testing.assert_allclose(affine.values_to_flux(values), expected)
    assert affine.flux_to_value(affine.value_to_flux(0.25)) == pytest.approx(0.25)


def test_flux_affine_rejects_zero_period() -> None:
    with pytest.raises(ValueError, match="flux_period"):
        FluxAffineMap(flux_half=0.0, flux_period=0.0)


def test_predict_dispersive_uses_fast_backend_with_provenance(monkeypatch) -> None:
    import zcu_tools.simulate.fluxonium.prediction as prediction_mod

    calls = []

    def fake_fast(
        params, fluxs, bare_rf, g, *, res_dim, qub_cutoff, qub_dim, return_dim
    ):
        calls.append(
            (params, np.asarray(fluxs), bare_rf, g, res_dim, qub_cutoff, qub_dim)
        )
        return tuple(np.full(len(fluxs), bare_rf + g + k) for k in range(return_dim))

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", fake_fast)

    resolution = PredictionResolution(qub_dim=7, qub_cutoff=11, res_dim=3)
    engine = FluxoniumPrediction((4.0, 1.0, 0.5), resolution=resolution)
    fluxs = np.array([0.1, 0.2], dtype=np.float64)
    result = engine.predict_dispersive(fluxs, g=0.06, bare_rf=5.3, return_dim=2)

    assert result.backend == "fast"
    assert not result.used_fallback
    assert len(result.lines) == 2
    assert calls[0][4:] == (3, 11, 7)
    np.testing.assert_allclose(calls[0][1], fluxs)


def test_predict_dispersive_fallback_exposes_provenance(monkeypatch) -> None:
    import zcu_tools.simulate.fluxonium.prediction as prediction_mod

    def boom(*args, **kwargs):
        raise DressedLabelingError("ambiguous")

    fallback_calls = []

    def fake_scqubits(
        params, fluxs, bare_rf, g, *, progress, res_dim, qub_cutoff, qub_dim, return_dim
    ):
        fallback_calls.append((progress, res_dim, qub_cutoff, qub_dim))
        return tuple(np.zeros(len(fluxs), dtype=np.float64) for _ in range(return_dim))

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", boom)
    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux", fake_scqubits)

    engine = FluxoniumPrediction((4.0, 1.0, 0.5))
    result = engine.predict_dispersive(
        np.array([0.3], dtype=np.float64), g=0.5, bare_rf=5.3
    )

    assert result.backend == "scqubits"
    assert result.used_fallback
    assert fallback_calls == [(False, 4, 30, 15)]


def test_axis_bound_session_caches_by_slider_args_and_owns_axis(monkeypatch) -> None:
    import zcu_tools.simulate.fluxonium.prediction as prediction_mod

    seen_fluxs: list[np.ndarray] = []

    def fake_fast(
        params, fluxs, bare_rf, g, *, res_dim, qub_cutoff, qub_dim, return_dim
    ):
        seen_fluxs.append(np.asarray(fluxs).copy())
        return tuple(np.full(len(fluxs), g + k) for k in range(return_dim))

    monkeypatch.setattr(prediction_mod, "calculate_dispersive_vs_flux_fast", fake_fast)

    axis = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    session = FluxoniumPrediction((4.0, 1.0, 0.5)).bind_flux_axis(axis)
    axis[0] = 9.9

    first = session.predict_dispersive(g=0.06, bare_rf=5.3, return_dim=2)
    second = session.predict_dispersive(g=0.06, bare_rf=5.3, return_dim=2)
    third = session.predict_dispersive(g=0.07, bare_rf=5.3, return_dim=2)

    assert first is second
    assert third is not first
    assert len(seen_fluxs) == 2
    np.testing.assert_allclose(seen_fluxs[0], np.array([0.1, 0.2, 0.3]))
    assert not session.flux_axis().flags.writeable


def test_frequency_curves_use_engine_affine_and_shared_sweep(monkeypatch) -> None:
    from zcu_tools.simulate.fluxonium import energies as energies_mod

    seen = {}

    def fake_calc(params, fluxs, cutoff=40, evals_count=20, spectrum_data=None):
        seen["fluxs"] = np.asarray(fluxs, dtype=np.float64)
        levels = np.arange(evals_count, dtype=np.float64)
        energies = seen["fluxs"][:, None] + levels[None, :] * 0.001
        return object(), energies

    monkeypatch.setattr(energies_mod, "calculate_energy_vs_flux", fake_calc)

    engine = FluxoniumPrediction(
        (4.0, 1.0, 0.5),
        flux_half=0.1,
        flux_period=0.8,
        flux_bias=0.03,
    )
    values = np.array([-0.2, 0.0, 0.3], dtype=np.float64)
    fluxs, freqs = engine.predict_frequencies_mhz(values, ((0, 1), (1, 3)))

    expected_fluxs = (values + 0.03 - 0.1) / 0.8 + 0.5
    np.testing.assert_allclose(fluxs, expected_fluxs)
    np.testing.assert_allclose(seen["fluxs"], expected_fluxs)
    np.testing.assert_allclose(freqs[0], np.ones(len(values)))
    np.testing.assert_allclose(freqs[1], np.full(len(values), 2.0))
