"""Observable running-sum and significance contracts for chain sweeps."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import norm, wishart
from zcu_tools.analysis.readout_chain import SweepSums, sweep_curve


def test_augmented_sums_preserve_counts_features_and_pool_products() -> None:
    augmented = np.zeros((3, 12, 12))
    augmented[:, 0, 0] = [2, 3, 4]
    features = np.arange(33.0).reshape(3, 11) / 1000
    augmented[:, 0, 1:] = features
    augmented[:, 1:, 0] = features
    augmented[:, 1:, 1:] = np.array([2, 3, 4])[:, None, None] * np.eye(11) + np.array(
        [1, 2, 3]
    )[:, None, None] * np.ones((11, 11))
    axis = np.array([5.0, 4.0, 6.0])

    sums = SweepSums.from_augmented(axis, augmented)

    assert_array_equal(sums.axis, [5, 4, 6])
    assert_array_equal(sums.dwell_count, [2, 3, 4])
    assert_array_equal(sums.feature_sums, features)
    assert_array_equal(sums.product_sum, 9 * np.eye(11) + 6 * np.ones((11, 11)))


def controlled_sums(
    means: np.ndarray, counts: np.ndarray, covariance: np.ndarray
) -> SweepSums:
    """Construct saved moments with the requested within-point covariance."""
    feature_sums = counts[:, None] * means
    product_sum = (counts.sum() - len(counts)) * covariance
    product_sum += means.T @ feature_sums
    return SweepSums(
        axis=np.arange(len(counts), dtype=float),
        dwell_count=counts,
        feature_sums=feature_sums,
        product_sum=product_sum,
    )


@pytest.mark.parametrize("column", [0, 7], ids=["IQ-line", "cross-only-line"])
def test_planted_line_is_the_curve_maximum_above_family_level(column: int) -> None:
    counts = np.linspace(1000, 2000, 101)
    lower = np.eye(11) + np.tril(np.full((11, 11), 0.15), -1)
    means = np.tile(np.arange(11) * 0.2, (101, 1))
    means[73] += 8 * lower[:, column] / np.sqrt(counts[73])
    sums = controlled_sums(means, counts, lower @ lower.T)

    curve = sweep_curve(sums)

    assert np.argmax(curve.sigma) == 73
    assert curve.sigma[73] > curve.level > 3
    assert_array_equal(curve.dwell_count, counts)
    np.testing.assert_allclose(curve.iq_mean, means[:, 0] + 1j * means[:, 1])
    if column >= 2:
        assert_array_equal(curve.iq_mean, np.full(101, 0.2j))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"dwell_count": np.array([1.0, 1000, 1000])}, "dwell"),
        ({"axis": np.arange(2.0)}, "shape"),
        ({"axis": np.empty(0)}, "shape"),
        ({"dwell_count": np.ones((3, 1))}, "shape"),
        ({"feature_sums": np.zeros((3, 10))}, "shape"),
        ({"product_sum": np.eye(10)}, "shape"),
        ({"axis": np.array([0.0, np.nan, 2.0])}, "finite"),
        ({"dwell_count": np.array([1000, np.inf, 1000])}, "finite"),
        ({"feature_sums": np.full((3, 11), np.nan)}, "finite"),
        ({"product_sum": np.full((11, 11), np.inf)}, "finite"),
        ({"product_sum": -np.eye(11)}, "positive definite"),
        ({"product_sum": np.eye(11) + np.triu(np.ones((11, 11)), 1)}, "symmetric"),
    ],
)
def test_invalid_saved_record_fails_at_entry(
    changes: dict[str, object], message: str
) -> None:
    valid = controlled_sums(np.zeros((3, 11)), np.full(3, 1000.0), np.eye(11))
    with pytest.raises(ValueError, match=message):
        sweep_curve(replace(valid, **changes))


@pytest.mark.parametrize(
    "augmented",
    [np.zeros((3, 11, 11)), np.zeros((2, 12, 12)), np.full((3, 12, 12), np.nan)],
)
def test_invalid_augmented_record_fails_before_pooling(augmented: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape|finite"):
        SweepSums.from_augmented(np.arange(3.0), augmented)


def test_gaussian_dwell_noise_has_calibrated_null_rates() -> None:
    # Exact Gaussian sufficient-statistic sampling avoids retaining 120M dwells:
    # sample means and pooled within-point Wishart scatter are independent.
    rng = np.random.default_rng(1024)
    points = 80000
    counts = rng.integers(1000, 2001, points).astype(float)
    lower = np.eye(11) + np.tril(np.full((11, 11), 0.2), -1)
    covariance = lower @ lower.T
    means = (rng.standard_normal((points, 11)) @ lower.T) / np.sqrt(counts[:, None])
    sums = controlled_sums(means, counts, covariance)
    residual = wishart.rvs(df=counts.sum() - points, scale=covariance, random_state=rng)
    sums = replace(sums, product_sum=residual + means.T @ sums.feature_sums)

    curve = sweep_curve(sums)

    for probability in (0.05, 0.01):
        expected = points * probability
        observed = np.count_nonzero(curve.sigma > norm.isf(probability / 2))
        assert abs(observed - expected) < 4 * np.sqrt(expected * (1 - probability))
    np.testing.assert_allclose(curve.level, norm.isf(0.05 / (2 * points)))


def test_independent_chi_square_blocks_have_calibrated_null_rates() -> None:
    rng = np.random.default_rng(920)
    draws = 400000
    # Antithetic means set the componentwise background to exactly zero without
    # changing any block distance. There are draws independent null statistics.
    distances = np.column_stack((rng.chisquare(2, draws), rng.chisquare(9, draws)))
    means = np.zeros((2 * draws, 11))
    means[:draws, 0] = np.sqrt(distances[:, 0] / 1000)
    means[:draws, 2] = np.sqrt(distances[:, 1] / 1000)
    means[draws:] = -means[:draws]
    sums = controlled_sums(means, np.full(2 * draws, 1000.0), np.eye(11))

    sigma = sweep_curve(sums).sigma[:draws]

    for probability in (0.05, 0.01):
        observed = np.count_nonzero(sigma > norm.isf(probability / 2))
        expected = draws * probability
        assert abs(observed - expected) < 4 * np.sqrt(expected * (1 - probability))


@pytest.mark.parametrize("field", ["dwell_count", "feature_sums", "axis"])
def test_curve_rejects_a_saved_array_changed_after_construction(field: str) -> None:
    sums = controlled_sums(np.zeros((3, 11)), np.full(3, 1000.0), np.eye(11))
    if field == "dwell_count":
        sums.dwell_count[0] = 1
    elif field == "feature_sums":
        sums.feature_sums[0, 0] = np.nan
    else:
        sums.axis.resize(2, refcheck=False)

    with pytest.raises(ValueError, match="dwell|finite|shape"):
        sweep_curve(sums)


def test_zero_evidence_is_zero_sigma_and_extreme_lines_remain_finite() -> None:
    means = np.zeros((101, 11))
    means[10, 0] = 2.0
    means[20, 0] = 3.0
    sums = controlled_sums(means, np.full(101, 1000.0), np.eye(11))

    curve = sweep_curve(sums)

    assert_array_equal(curve.sigma[30:], np.zeros(71))
    assert np.isfinite(curve.sigma).all()
    assert curve.sigma[20] > curve.sigma[10] > 38  # p far below 1e-300
