"""Public squared-rate sine/cosine regression and uncertainty propagation."""

from importlib import import_module

import numpy as np
import pytest


def test_maximum_phase_and_covariance_recover_known_coefficients():
    owner = import_module("zcu_tools.analysis.fitting")
    phases = np.arange(0, 360, 45, dtype=np.float64)
    # Orthogonal second harmonic leaves A=5, B=3, C=4 exactly identifiable.
    radians = np.deg2rad(phases)
    squared = 5 + 3 * np.cos(radians) + 4 * np.sin(radians) + 0.2 * np.cos(2 * radians)
    result = owner.fit_phase_modulation(phases, squared)
    np.testing.assert_allclose(result.coefficients, [5, 3, 4], atol=1e-12)
    np.testing.assert_allclose(
        result.covariance, np.diag([0.004, 0.008, 0.008]), atol=1e-14
    )
    prediction = result.prediction
    assert prediction is not None
    assert prediction.maximum_phase_deg == pytest.approx(53.13010235415598)
    # atan2 gradient norm²=1/(3²+4²), covariance_BC=0.008 I.
    assert result.phase_variance_rad2 == pytest.approx(0.00032)
    assert result.phase_error_deg == pytest.approx(np.rad2deg(np.sqrt(0.00032)))
    assert result.reason is None
    assert result.fit_quality is not None
    assert prediction.squared_rates.shape == prediction.phase_degrees.shape == (721,)
    assert np.max(prediction.squared_rates) == pytest.approx(10, abs=1e-4)


def test_rank_deficient_phase_coverage_is_unavailable():
    owner = import_module("zcu_tools.analysis.fitting")
    result = owner.fit_phase_modulation(np.zeros(5), np.arange(5, dtype=np.float64))
    assert result.prediction is None
    assert result.coefficients is None and result.covariance is None
    assert "rank deficient" in result.reason


def test_zero_modulation_has_quality_but_no_defined_phase():
    owner = import_module("zcu_tools.analysis.fitting")
    result = owner.fit_phase_modulation(
        np.arange(0, 360, 45, dtype=np.float64), np.full(8, 5.0)
    )
    assert result.prediction is None
    assert result.coefficients is not None and result.covariance is not None
    assert result.fit_quality is not None
    assert "Zero rate² modulation" in result.reason


def test_three_slices_cannot_estimate_residual_scaled_uncertainty():
    owner = import_module("zcu_tools.analysis.fitting")
    result = owner.fit_phase_modulation(
        np.array([0.0, 90.0, 180.0]), np.array([5.0, 6.0, 4.0])
    )
    assert result.prediction is None
    assert "more than three" in result.reason
