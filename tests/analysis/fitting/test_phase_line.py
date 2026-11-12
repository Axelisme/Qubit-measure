"""Public degree-phase unwrap and count-weighted straight-line contract."""

from importlib import import_module

import numpy as np
import pytest


def test_unwrap_crosses_turns_and_recovers_weighted_line_covariance():
    owner = import_module("zcu_tools.analysis.fitting")
    n = np.arange(5, dtype=np.float64)
    phases = np.array([160.0, -125.0, -50.0, 25.0, 100.0])
    result = owner.fit_phase_line(n, phases, np.array([1.0, 4.0, 1.0, 4.0, 1.0]))
    assert result.reason is None
    assert result.intercept_deg == pytest.approx(160)
    assert result.slope_deg == pytest.approx(75)
    np.testing.assert_allclose(result.unwrapped_deg, [160, 235, 310, 385, 460])
    np.testing.assert_allclose(result.fit_curve, [160, 235, 310, 385, 460])
    # Known absolute variances, not residual-rescaled despite exact fit.
    np.testing.assert_allclose(
        result.covariance, [[90 / 119, -4 / 17], [-4 / 17, 2 / 17]]
    )
    assert result.slope_error_deg == pytest.approx(np.sqrt(2 / 17))
    assert result.fit_quality is not None
    assert result.fit_quality.r2 == pytest.approx(1)


def test_adjacent_half_turn_is_an_explicit_unwrap_ambiguity():
    owner = import_module("zcu_tools.analysis.fitting")
    result = owner.fit_phase_line(
        np.arange(3, dtype=np.float64), np.array([10.0, -170.0, 0.0]), np.ones(3)
    )
    assert result.slope_deg is None
    assert "half-turn" in result.reason


@pytest.mark.parametrize(
    "variances", [np.array([1.0, 0.0, 1.0]), np.array([1.0, np.nan, 1.0])]
)
def test_nonpositive_or_nonfinite_variance_leaves_no_slope(variances):
    owner = import_module("zcu_tools.analysis.fitting")
    result = owner.fit_phase_line(
        np.arange(3, dtype=np.float64), np.array([10.0, 20.0, 30.0]), variances
    )
    assert result.slope_deg is None
    assert "variance" in result.reason


def test_repeated_coordinate_cannot_identify_line():
    owner = import_module("zcu_tools.analysis.fitting")
    result = owner.fit_phase_line(np.ones(3), np.array([10.0, 20.0, 30.0]), np.ones(3))
    assert result.slope_deg is None
    assert "Unidentifiable" in result.reason
