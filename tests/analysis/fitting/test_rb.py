"""Public RB projection, seed statistics and decay fit contracts."""

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


def test_recovers_two_decays_on_one_iq_axis_and_discards_incomplete_pairs() -> None:
    depths = np.array([1, 4, 9, 16, 25, 40], dtype=np.int64)
    expected = np.array([0.98, 0.96])
    curves = np.array([0.4 * p**depths + 0.3 for p in expected])
    signals = (
        np.broadcast_to(curves * (0.6 + 0.8j) + (2 - 1j), (4, 2, 6))
        .copy()
        .astype(np.complex128)
    )
    signals[1] += 0.1 * (0.6 + 0.8j)
    signals[2] -= 0.1 * (0.6 + 0.8j)
    signals[0, 1, 2] = np.nan
    signals[3] = np.nan
    # A rejected pair cannot distort either arm or the common IQ direction.
    signals[0, 0] = 1e6 + 3e6j
    fit = owner.analyze_irb(depths, signals, bootstrap_samples=100, bootstrap_seed=17)
    assert fit.summary.p_reference == pytest.approx(0.98, abs=2e-7)
    assert fit.summary.p_interleaved == pytest.approx(0.96, abs=2e-7)
    assert fit.summary.n_paired_seeds == 2
    assert fit.summary.bootstrap_successes == 100
    # Unit IQ direction gives a baseline projection of 0.4, not abs(IQ).
    np.testing.assert_allclose(fit.mean_signals, curves + 0.4, atol=1e-14)
    np.testing.assert_allclose(fit.fit_signals, curves + 0.4, atol=1e-7)
    fidelity = (1 + 0.96 / 0.98) / 2
    assert fit.summary.gate_fidelity == pytest.approx(fidelity, abs=2e-7)
    assert fit.summary.fidelity_ci_low == pytest.approx(fidelity, abs=2e-7)
    assert fit.summary.fidelity_ci_high == pytest.approx(fidelity, abs=2e-7)


def test_reference_decay_weights_complete_seeds_equally_and_bootstraps_epc() -> None:
    depths = np.array([1, 4, 9, 16, 25, 40], dtype=np.int64)
    slow = 0.4 * 0.98**depths + 0.3
    fast = 0.4 * 0.94**depths + 0.3
    signals = np.full((3, 1, len(depths)), np.nan, dtype=np.complex128)
    direction = 0.6 + 0.8j
    signals[0, 0] = slow * direction
    signals[1, 0] = fast * direction
    signals[2, 0, 0] = 100j  # Incomplete seed must not distort the projection.

    fit = owner.analyze_reference_rb(depths, signals, bootstrap_samples=100)

    np.testing.assert_allclose(fit.mean_signals, [(slow + fast) / 2], atol=1e-14)
    assert fit.fit_signals.shape == (1, len(depths))
    assert 0.94 < fit.summary.p_reference < 0.98
    assert fit.summary.reference_epc == pytest.approx((1 - fit.summary.p_reference) / 2)
    # With two seeds, each homogeneous resample has 25% probability. The 95%
    # interval reaches the per-seed EPCs; resampling depths would differ.
    assert fit.summary.epc_ci_low == pytest.approx(0.01, abs=2e-7)
    assert fit.summary.epc_ci_high == pytest.approx(0.03, abs=2e-7)
    assert fit.summary.n_seeds == 2
    assert fit.summary.bootstrap_successes == 100


def test_seed_bootstrap_is_reproducible_and_retains_negative_error_ci() -> None:
    depths = np.array([1, 4, 9, 16, 25, 40], dtype=np.int64)
    rng = np.random.default_rng(307)
    curves = np.array([0.4 * p**depths + 0.3 for p in (0.96, 0.98)])
    signals = np.asarray(
        curves[None] + rng.normal(0, 0.0008, (5, 2, 6)),
        dtype=np.complex128,
    )
    first = owner.analyze_irb(depths, signals, bootstrap_samples=100, bootstrap_seed=31)
    second = owner.analyze_irb(
        depths, signals, bootstrap_samples=100, bootstrap_seed=31
    )
    assert first.summary == second.summary
    assert first.summary.bootstrap_successes == 100
    assert first.summary.gate_error < 0
    assert first.summary.gate_fidelity > 1
    assert 1 < first.summary.fidelity_ci_low < first.summary.fidelity_ci_high
    assert first.summary.fidelity_ci_high - first.summary.fidelity_ci_low < 0.002
    assert (
        first.summary.fidelity_ci_low
        < (1 + 0.98 / 0.96) / 2
        < first.summary.fidelity_ci_high
    )
    assert "negative error estimate retained" in first.summary.warning


@pytest.mark.parametrize(
    "problem", ["shape", "depths", "bootstrap", "seeds", "contrast"]
)
def test_unidentifiable_or_incomplete_inputs_fail_explicitly(problem: str) -> None:
    depths = np.array([1, 4, 9, 16], dtype=np.int64)
    signals = np.broadcast_to(np.array([0.8, 0.7, 0.6, 0.5]), (2, 2, 4)).astype(
        np.complex128
    )
    samples = 100
    if problem == "shape":
        signals = signals[None]
    elif problem == "depths":
        depths[:] = 1
    elif problem == "bootstrap":
        samples = 99
    elif problem == "seeds":
        signals[1] = np.nan
    else:
        signals[:] = 0.5
    with pytest.raises(
        ValueError, match="Expected IQ shape|Need at least|no measurable contrast"
    ):
        owner.analyze_irb(depths, signals, bootstrap_samples=samples)


def test_live_projection_preserves_a_single_reference_arm() -> None:
    signals = np.full((2, 1, 4), np.nan, dtype=np.complex128)
    signals[0, 0, :2] = [-3 - 4j, -1.8 - 2.4j]
    signals[1, 0, [0, 3]] = [-4.2 - 5.6j, -0.6 - 0.8j]

    means = owner.project_arm_means(signals).mean_signals

    np.testing.assert_allclose(means, [[6, 3, np.nan, 1]], atol=1e-14)


def test_live_projection_uses_available_samples_one_direction_and_nan_slots() -> None:
    signals = np.full((2, 2, 3), np.nan, dtype=np.complex128)
    # Negative I direction makes a magnitude or independent-arm projection wrong.
    signals[0, 0] = [-3 - 4j, -1.8 - 2.4j, -0.6 - 0.8j]
    signals[1, 1, :2] = [-2.4 - 3.2j, -1.2 - 1.6j]
    result = owner.project_arm_means(signals).mean_signals
    np.testing.assert_allclose(result[0], [5, 3, 1], atol=1e-14)
    np.testing.assert_allclose(result[1, :2], [4, 2], atol=1e-14)
    assert np.isnan(result[1, 2])
    empty = owner.project_arm_means(
        np.full((1, 2, 2), np.nan, dtype=np.complex128)
    ).mean_signals
    assert empty.shape == (2, 2)
    assert np.isnan(empty).all()
