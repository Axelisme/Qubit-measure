"""Public paired-seed statistics and shared-decay numeric contracts."""

from importlib import import_module

import numpy as np
import pytest

owner = import_module("zcu_tools.analysis.fitting")


def test_seed_statistics_keep_equal_seed_weights_and_binomial_floor() -> None:
    survival = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.25, 0.25], [0.75, 0.75]],
            [[0.99, 0.99], [0.01, np.nan]],
        ]
    )
    totals = np.broadcast_to(
        np.array([10.0, 20.0, 40.0])[:, None, None], survival.shape
    )
    statistics = owner.seed_statistics(survival, totals)
    np.testing.assert_array_equal(statistics.seed_means[:2], survival[:2])
    np.testing.assert_allclose(statistics.seed_variances[0], 0.0025)
    np.testing.assert_allclose(statistics.seed_variances[1], 0.009375)
    assert np.isnan(statistics.seed_means[2]).all()
    assert np.isnan(statistics.seed_variances[2]).all()
    np.testing.assert_array_equal(statistics.means, [[0.125, 0.625], [0.875, 0.375]])
    np.testing.assert_allclose(statistics.errors, np.sqrt(0.011875) / 2)


@pytest.mark.parametrize("total", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_statistics_reject_invalid_contributing_totals_but_ignore_missing_seeds(
    total: float,
) -> None:
    survival = np.full((4, 2, 2), 0.5)
    totals = np.full_like(survival, 100.0)
    totals[0, 0, 0] = total
    with pytest.raises(ValueError, match="totals"):
        owner.seed_statistics(survival, totals)

    survival[0, 1, 1] = np.nan
    survival[3] = np.nan
    totals[3] = total
    result = owner.seed_statistics(survival, totals)
    np.testing.assert_array_equal(result.means, np.full((2, 2), 0.5))
    np.testing.assert_allclose(result.errors, np.sqrt(0.005) / 2)
    assert np.isnan(result.seed_means[[0, 3]]).all()


@pytest.mark.parametrize("case", ["extra-arm", "broadcast-totals"])
def test_statistics_reject_shapes_that_would_silently_change_statistics(
    case: str,
) -> None:
    survival = np.full((2, 3 if case == "extra-arm" else 2, 2), 0.5)
    totals = np.full_like(survival, 100.0)
    if case == "broadcast-totals":
        totals = totals[..., :1]
    with pytest.raises(ValueError, match="shape"):
        owner.seed_statistics(survival, totals)


def test_repeated_seed_draws_retain_multiplicity_without_shot_weighting() -> None:
    survival = np.broadcast_to(
        np.array([0.2, 0.4, 0.8])[:, None, None], (3, 2, 2)
    ).copy()
    totals = np.full_like(survival, 10.0)
    draw = np.array([2, 2, 0])
    result = owner.seed_statistics(survival[draw], totals[draw])
    np.testing.assert_allclose(result.means, 0.6)
    np.testing.assert_allclose(result.errors, np.sqrt(0.048) / 3)
    with pytest.raises(ValueError, match="at least two seeds"):
        owner.seed_statistics(survival[:1], totals[:1])


@pytest.mark.parametrize("case", ["extra-group", "extra-arm", "extra-error-group"])
def test_shared_decay_rejects_shapes_that_would_leave_data_unused(case: str) -> None:
    depths = np.array([1, 4, 9, 16, 25, 40, 60, 80], dtype=np.int64)
    means = np.array(
        [
            [0.4 * 0.98**depths + 0.3, 0.35 * 0.96**depths + 0.2],
            [0.25 * 0.98**depths + 0.2, 0.3 * 0.96**depths + 0.35],
        ]
    )
    errors = np.full_like(means, 0.01)
    if case == "extra-group":
        means = np.concatenate([means, means[:1]])
        errors = np.concatenate([errors, errors[:1]])
    elif case == "extra-arm":
        means = np.concatenate([means, means[:, :1]], axis=1)
        errors = np.concatenate([errors, errors[:, :1]], axis=1)
    else:
        errors = np.concatenate([errors, errors[:1]])
    with pytest.raises(ValueError, match="shape"):
        owner.fit_joint_decay(depths, means, errors)


def test_shared_decay_recovers_both_arm_rates_and_full_covariance() -> None:
    depths = np.array([1, 4, 9, 16, 25, 40, 60, 80], dtype=np.int64)
    ps = (0.98, 0.96)
    amplitudes = ((0.4, 0.35), (0.25, 0.3))
    offsets = ((0.3, 0.2), (0.2, 0.35))
    exact = np.array(
        [
            [amplitudes[g][a] * ps[a] ** depths + offsets[g][a] for a in range(2)]
            for g in range(2)
        ]
    )
    noise = 0.0001 * np.sin(np.arange(32).reshape(2, 2, 8))
    means = exact + noise
    errors = np.full_like(means, 0.01)
    result = owner.fit_joint_decay(depths, means, errors)
    fit = result.fit
    assert fit.values["p_0"] == pytest.approx(0.98, abs=3e-5)
    assert fit.values["p_1"] == pytest.approx(0.96, abs=3e-5)
    np.testing.assert_allclose(result.curves, exact, atol=0.00015)
    assert fit.diagnostics.valid
    # Independent local linear error propagation for A*p**depth+B, with the
    # documented reduced-chi-square covariance scaling of shared fitting.
    jacobian = np.zeros((32, 10))
    names = fit.parameter_names
    for g in range(2):
        for a in range(2):
            rows = slice((g * 2 + a) * 8, (g * 2 + a + 1) * 8)
            jacobian[rows, names.index(f"p_{a}")] = (
                amplitudes[g][a] * depths * ps[a] ** (depths - 1)
            )
            jacobian[rows, names.index(f"A_{g}_{a}")] = ps[a] ** depths
            jacobian[rows, names.index(f"B_{g}_{a}")] = 1
    expected_cov = np.linalg.inv(jacobian.T @ jacobian / 0.01**2)
    expected_cov *= fit.diagnostics.reduced_chi_square
    np.testing.assert_allclose(fit.covariance, expected_cov, rtol=0.02, atol=1e-13)
    first, second = names.index("A_0_0"), names.index("A_1_0")
    assert abs(fit.covariance[first, second]) > 1e-11
