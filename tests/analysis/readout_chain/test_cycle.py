"""Saved cycle moments and live-curve contracts through the public interface."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import wishart
from zcu_tools.analysis.readout_chain import (
    CycleNuisances,
    CycleSums,
    cycle_curves,
    fit_cycle,
)

from ._exact_moments import curve_moments, repeated_curve


def test_fit_cycle_wald_errors_follow_axis_units() -> None:
    model = repeated_curve(
        lambda x, p: np.sin(np.pi * p["frequency"] * x) ** 2,
        ("frequency",),
        ((0.1, 2.0),),
        oscillation_frequency="frequency",
        derivative=lambda x, p: (np.pi * x * np.sin(2 * np.pi * p["frequency"] * x))[
            :, None
        ],
    )
    nuisance = CycleNuisances(
        np.array(
            [
                [[0.8, 0.15, 0.05], [0.2, 0.65, 0.15]],
                [[0.65, 0.25, 0.1], [0.05, 0.9, 0.05]],
                [[0.05, 0.05, 0.9], [0.15, 0.15, 0.7]],
            ]
        ),
        np.array([[-1.0, 0.0], [1.0, 0.2], [0.0, 1.8]]),
        0.25,
        np.array([0.665, 0.285, 0.05]),
        np.array([0.95, 0.8, 0.95]),
        np.array([0.63, 0.27, 0.1]),
    )
    axis = np.linspace(0.05, 3.0, 9)
    moments = curve_moments(model, axis, {"frequency": 1.2}, nuisance)
    sums = CycleSums(
        axis, np.full(len(axis), 3000.0), 3000 * moments.means, np.zeros((65, 65))
    )
    original = fit_cycle(sums, model, {"frequency": 1.1})

    # Rescaling time coordinates changes the frequency by the reciprocal factor.
    unit = 1e-7
    converted = fit_cycle(
        replace(sums, axis=axis * unit),
        replace(model, bounds=((0.1 / unit, 2.0 / unit),)),
        {"frequency": 1.1 / unit},
    )

    assert original.errors["frequency"] > 0
    assert converted.parameters["frequency"] * unit == pytest.approx(
        original.parameters["frequency"], rel=1e-5
    )
    assert converted.errors["frequency"] * unit == pytest.approx(
        original.errors["frequency"], rel=1e-4
    )
    assert converted.reliability == pytest.approx(original.reliability, rel=1e-4)


def test_augmented_sums_keep_point_counts_and_features_but_pool_products() -> None:
    axis = np.array([3.0, 1.0, 2.0])
    augmented = np.zeros((3, 66, 66))
    augmented[:, 0, 0] = [2, 3, 4]
    features = np.arange(195.0).reshape(3, 65)
    augmented[:, 0, 1:] = features
    augmented[:, 1:, 0] = features
    augmented[:, 1:, 1:] = np.array([2, 3, 4])[:, None, None] * np.eye(65)

    sums = CycleSums.from_augmented(axis, augmented)
    axis[:] = 0
    augmented[:] = 0

    assert_array_equal(sums.axis, [3, 1, 2])
    assert_array_equal(sums.cycle_count, [2, 3, 4])
    assert_array_equal(sums.feature_sums, features)
    assert_array_equal(sums.product_sum, 9 * np.eye(65))


def controlled_sums(
    means: np.ndarray, counts: np.ndarray, covariance: np.ndarray
) -> CycleSums:
    """Construct moments with a chosen within-point covariance."""
    feature_sums = counts[:, None] * means
    return CycleSums(
        axis=np.arange(len(counts), dtype=float),
        cycle_count=counts,
        feature_sums=feature_sums,
        product_sum=(counts.sum() - len(counts)) * covariance + means.T @ feature_sums,
    )


def test_planted_shift_uses_all_features_and_aligns_first_read_in_sigma() -> None:
    shift = np.array([-3.0, -1.0, 0.0, 1.0, 6.0])
    counts = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    means = np.full((5, 65), 2.0)
    means[:, 0] += 2 * shift
    means[:, 64] += 3 * shift
    covariance = np.eye(65)
    covariance[0, 0] = 4
    covariance[1, 1] = 9
    sums = controlled_sums(means, counts, covariance)

    curves = cycle_curves(sums)

    # A unit shift is one first-read sigma and sqrt(1 + 3**2) joint sigmas.
    orientation = np.sign(curves.first_read_sigma @ shift)
    np.testing.assert_allclose(
        curves.first_read_sigma, orientation * shift * np.sqrt(counts), atol=1e-10
    )
    np.testing.assert_allclose(
        curves.main_sigma, np.sqrt(10) * curves.first_read_sigma, atol=1e-10
    )
    assert np.corrcoef(curves.main_sigma, curves.first_read_sigma)[0, 1] > 0.999
    assert_array_equal(curves.cycle_count, counts)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"cycle_count": np.array([1.0, 100.0, 100.0])}, "cycle"),
        ({"axis": np.arange(2.0)}, "shape"),
        ({"axis": np.empty(0)}, "shape"),
        ({"cycle_count": np.ones((3, 1))}, "shape"),
        ({"feature_sums": np.zeros((3, 64))}, "shape"),
        ({"product_sum": np.eye(64)}, "shape"),
        ({"axis": np.array([0.0, np.nan, 2.0])}, "finite"),
        ({"cycle_count": np.array([100.0, np.inf, 100.0])}, "finite"),
        ({"feature_sums": np.full((3, 65), np.nan)}, "finite"),
        ({"product_sum": np.full((65, 65), np.inf)}, "finite"),
    ],
)
def test_invalid_saved_record_is_rejected_at_construction(
    changes: dict[str, object], message: str
) -> None:
    sums = controlled_sums(np.zeros((3, 65)), np.full(3, 100.0), np.eye(65))

    with pytest.raises(ValueError, match=message):
        replace(sums, **changes)


@pytest.mark.parametrize(
    "augmented",
    [
        np.zeros(3),
        np.zeros((3, 65, 65)),
        np.zeros((2, 66, 66)),
        np.zeros((3, 66, 67)),
        # Non-finite mirrored sums must fail even though pooling uses row zero.
        np.tile(2 * np.eye(66), (3, 1, 1))
        + np.pad(np.full((3, 1, 1), np.nan), ((0, 0), (1, 64), (0, 65))),
    ],
)
def test_malformed_augmented_arrays_raise_before_pooling(augmented: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape|finite"):
        CycleSums.from_augmented(np.arange(3.0), augmented)


@pytest.mark.parametrize(
    "field", ["axis", "cycle_count", "feature_sums", "product_sum"]
)
def test_curves_reject_arrays_changed_after_record_construction(field: str) -> None:
    sums = controlled_sums(np.zeros((3, 65)), np.full(3, 100.0), np.eye(65))
    if field == "axis":
        sums.axis.resize(2, refcheck=False)
    elif field == "cycle_count":
        sums.cycle_count[0] = 1
    elif field == "feature_sums":
        sums.feature_sums[0, 0] = np.nan
    else:
        sums.product_sum.resize((64, 64), refcheck=False)

    with pytest.raises(ValueError, match="shape|cycle|finite"):
        cycle_curves(sums)


@pytest.mark.parametrize("direction", [[-2.0, 1.0], [1.0, -2.0]])
def test_first_read_axis_has_positive_dominant_component(
    direction: list[float],
) -> None:
    means = np.zeros((5, 65))
    shift = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    means[:, :2] = shift[:, None] * direction
    sums = controlled_sums(means, np.full(5, 100.0), np.eye(65))

    curves = cycle_curves(sums)

    # The principal axis is opposite to direction: its dominant component is positive.
    np.testing.assert_allclose(curves.first_read_sigma, -10 * np.sqrt(5) * shift)


def test_first_read_axis_is_selected_before_noise_normalization() -> None:
    means = np.zeros((4, 65))
    means[:, :2] = [[4, 0], [-4, 0], [0, 2], [0, -2]]
    covariance = np.eye(65)
    covariance[0, 0] = 100
    sums = controlled_sums(means, np.full(4, 100.0), covariance)

    curves = cycle_curves(sums)

    # Raw IQ varies most along I; after whitening, Q is the main direction.
    np.testing.assert_allclose(np.abs(curves.first_read_sigma), [4, 4, 0, 0])
    np.testing.assert_allclose(np.abs(curves.main_sigma), [0, 0, 20, 20])


@pytest.mark.parametrize("variance", [0.0, -1.0])
def test_non_positive_definite_covariance_is_not_regularized(variance: float) -> None:
    covariance = np.eye(65)
    covariance[64, 64] = variance
    sums = controlled_sums(np.zeros((3, 65)), np.full(3, 100.0), covariance)

    with pytest.raises(ValueError, match="positive definite"):
        cycle_curves(sums)


def test_constant_point_means_produce_two_finite_zero_curves() -> None:
    sums = controlled_sums(np.full((3, 65), 2.0), np.full(3, 100.0), np.eye(65))

    curves = cycle_curves(sums)

    assert_array_equal(curves.main_sigma, np.zeros(3))
    assert_array_equal(curves.first_read_sigma, np.zeros(3))


def test_gaussian_cycle_noise_is_scaled_to_sigma_units() -> None:
    rng = np.random.default_rng(614)
    points = 50000
    counts = rng.integers(1000, 2001, points).astype(float)
    lower = np.eye(65) + np.tril(np.full((65, 65), 0.1), -1)
    covariance = lower @ lower.T
    means = (rng.standard_normal((points, 65)) @ lower.T) / np.sqrt(counts[:, None])
    sums = controlled_sums(means, counts, covariance)
    # Gaussian means and within-point Wishart scatter are independent sufficient
    # statistics, so no raw cycles or acquisition processor are needed.
    residual = wishart.rvs(df=counts.sum() - points, scale=covariance, random_state=rng)
    sums = replace(sums, product_sum=residual + means.T @ sums.feature_sums)

    curves = cycle_curves(sums)

    # Fitting the direction to these same points selects the largest sample
    # eigenvalue: allow its O(sqrt(dimension / points)) inflation plus scatter.
    for curve, dimension in ((curves.main_sigma, 65), (curves.first_read_sigma, 2)):
        tolerance = 2 * np.sqrt(dimension / points) + 5 * np.sqrt(2 / points)
        assert abs(np.var(curve, ddof=1) - 1) < tolerance
    assert np.corrcoef(curves.main_sigma, curves.first_read_sigma)[0, 1] >= 0
