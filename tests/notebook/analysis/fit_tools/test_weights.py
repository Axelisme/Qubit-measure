from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.notebook.analysis.fit_tools import (
    FluxResidualWeighting,
    MeasurementErrorPolicy,
    build_flux_residual_weights,
    least_squares_cost,
    resolve_measurement_errors,
)


def test_equal_flux_bin_weights_normalize_each_bin() -> None:
    fluxs = np.array([0.490, 0.491, 0.492, 0.500, 0.501], dtype=np.float64)

    weights = build_flux_residual_weights(
        fluxs,
        FluxResidualWeighting(mode="equal_flux_bin", bin_width=0.005, origin=0.49),
    )

    np.testing.assert_array_equal(weights.bin_counts, [3, 3, 3, 2, 2])
    np.testing.assert_allclose(
        weights.residual_weights,
        [
            1 / np.sqrt(3),
            1 / np.sqrt(3),
            1 / np.sqrt(3),
            1 / np.sqrt(2),
            1 / np.sqrt(2),
        ],
    )
    assert weights.effective_observation_count == 2.0


def test_equal_flux_bin_weights_equalize_linear_least_squares_cost() -> None:
    dense_fluxs = np.full(100, 0.49, dtype=np.float64)
    sparse_fluxs = np.array([0.51], dtype=np.float64)
    fluxs = np.r_[dense_fluxs, sparse_fluxs]
    raw_residuals = np.full_like(fluxs, 10.0)

    weights = build_flux_residual_weights(
        fluxs,
        FluxResidualWeighting(mode="equal_flux_bin", bin_width=0.005, origin=0.49),
    )
    weighted = raw_residuals * weights.residual_weights

    assert least_squares_cost(weighted[:100]) == pytest.approx(
        least_squares_cost(weighted[100:])
    )


def test_bin_median_error_fill_uses_local_bin_then_global() -> None:
    values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    errors = np.array([1.0, np.nan, np.nan, 8.0], dtype=np.float64)
    fluxs = np.array([0.490, 0.491, 0.510, 0.520], dtype=np.float64)
    weights = build_flux_residual_weights(
        fluxs,
        FluxResidualWeighting(mode="equal_flux_bin", bin_width=0.005, origin=0.49),
    )

    resolved = resolve_measurement_errors(
        values,
        errors,
        policy=MeasurementErrorPolicy(nan_policy="bin_median"),
        flux_weights=weights,
    )

    np.testing.assert_allclose(resolved.effective_errors, [1.0, 1.0, 4.5, 8.0])
    np.testing.assert_array_equal(resolved.bin_fill_mask, [False, True, False, False])
    np.testing.assert_array_equal(
        resolved.global_fill_mask, [False, False, True, False]
    )


def test_error_floor_applies_to_measured_and_filled_errors() -> None:
    values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    errors = np.array([0.1, np.nan, 10.0], dtype=np.float64)
    weights = build_flux_residual_weights(
        np.array([0.49, 0.49, 0.50], dtype=np.float64),
        FluxResidualWeighting(mode="equal_flux_bin", bin_width=0.01, origin=0.49),
    )

    resolved = resolve_measurement_errors(
        values,
        errors,
        policy=MeasurementErrorPolicy(
            nan_policy="bin_median",
            absolute_floor=0.2,
            relative_floor=0.05,
        ),
        flux_weights=weights,
    )

    np.testing.assert_allclose(resolved.effective_errors, [0.5, 1.0, 10.0])
    np.testing.assert_array_equal(resolved.floor_mask, [True, True, False])


def test_bin_median_requires_flux_weights() -> None:
    with pytest.raises(ValueError, match="flux_weights"):
        resolve_measurement_errors(
            np.array([1.0], dtype=np.float64),
            np.array([np.nan], dtype=np.float64),
            policy=MeasurementErrorPolicy(nan_policy="bin_median", fallback_error=1.0),
        )
