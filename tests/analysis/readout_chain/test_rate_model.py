"""Detailed-balance T1 fits at the public fit_cycle and model seams."""

import numpy as np
import pytest
from scipy.linalg import expm
from zcu_tools.analysis.readout_chain import (
    CycleFit,
    CycleSums,
    RateNuisances,
    cycle_curves,
    cycle_residual_projection,
    fit_cycle,
    t1_model,
)

from ._exact_moments import rate_moments
from ._rate_truths import ALL_PAIRS, LADDER, PI_SWAP, STAR, THERMAL, RateTruth


def expect_rate_nuisances(fit: CycleFit) -> RateNuisances:
    if not isinstance(fit.nuisances, RateNuisances):
        raise AssertionError("rate fit must return rate nuisances")
    return fit.nuisances


@pytest.mark.parametrize("truth", [STAR, LADDER, ALL_PAIRS], ids=lambda t: t.name)
def test_exact_moments_recover_both_modes_and_larger_amplitude_t1(
    truth: RateTruth,
) -> None:
    sums = truth.sums()
    result = fit_cycle(sums, t1_model(), {"t1": 20.0})
    relaxation = result.relaxation
    assert relaxation is not None
    times, amplitudes = truth.modes()
    winner = int(np.argmax(amplitudes))
    np.testing.assert_allclose(relaxation.eigen_times, times, rtol=1e-5)
    np.testing.assert_allclose(relaxation.amplitudes, amplitudes, rtol=1e-5)
    np.testing.assert_allclose(relaxation.t1, times[winner], rtol=1e-5)
    np.testing.assert_allclose(relaxation.secondary_time, times[1 - winner], rtol=1e-5)
    assert relaxation.t1_error > 0
    assert relaxation.secondary_error > 0
    np.testing.assert_allclose(
        relaxation.eigen_time_errors, oracle_time_errors(result), rtol=1e-6
    )
    assert result.reliability is None
    assert result.J < 1e-7
    assert result.df == 13 * 65 - 23
    assert result.j_per_df == pytest.approx(result.J / 822)
    assert result.expected_band == pytest.approx((0.9013473, 1.0986527))
    assert result.tail_fraction == 0.0
    np.testing.assert_allclose(
        result.predicted_means, sums.feature_sums / 3000, atol=1e-7
    )
    projected = cycle_residual_projection(
        sums, t1_model(), result, model_axis=sums.axis
    )
    np.testing.assert_allclose(
        projected.model_sigma, cycle_curves(sums).main_sigma, atol=1e-4
    )


def test_zero_delays_leave_rates_and_eigen_times_unidentified() -> None:
    axis = np.zeros(3)
    nuisance = coordinate_nuisance(oracle_coordinates())
    parameters: dict[str, float] = dict(
        zip(("rate_01", "rate_02", "rate_12"), (0.05, 0.08, 0.025), strict=True)
    )
    moments = rate_moments(axis, parameters, nuisance)
    sums = CycleSums(axis, np.full(3, 3000.0), 3000 * moments.means, np.zeros((65, 65)))

    result = fit_cycle(sums, t1_model(), {"t1": 20.0})

    assert all(np.isnan(error) for error in result.errors.values())
    relaxation = result.relaxation
    assert relaxation is not None
    assert np.isnan(relaxation.eigen_time_errors).all()
    assert np.isnan(relaxation.t1_error)
    assert np.isnan(relaxation.secondary_error)
    assert result.reliability is None


def oracle_time_errors(fit: CycleFit) -> np.ndarray:
    """Differentiate nonsymmetric generator eigen times, only in tests."""
    nuisance = expect_rate_nuisances(fit)
    pi = nuisance.thermal
    s = pi[0] + pi[1]
    values = np.array([*fit.parameters.values(), s, pi[0] / s])

    def times(v: np.ndarray) -> np.ndarray:
        thermal = np.array([v[3] * v[4], v[3] * (1 - v[4]), 1 - v[3]])
        generator = np.zeros((3, 3))
        for rate, (i, j) in zip(v[:3], ((0, 1), (0, 2), (1, 2)), strict=True):
            generator[i, j] = rate * thermal[j] / (thermal[i] + thermal[j])
            generator[j, i] = rate * thermal[i] / (thermal[i] + thermal[j])
        generator -= np.diag(generator.sum(axis=1))
        eigenvalues = np.sort(np.real(np.linalg.eigvals(generator)))[::-1]
        return -1 / eigenvalues[1:]

    gradient = np.zeros((2, len(fit.parameter_names)))
    columns = [
        fit.parameter_names.index(name)
        for name in ("rate_01", "rate_02", "rate_12", "thermal_s", "thermal_f")
    ]
    for i, col in enumerate(columns):
        plus, minus = values.copy(), values.copy()
        plus[i] += 1e-7
        minus[i] -= 1e-7
        gradient[:, col] = (times(plus) - times(minus)) / 2e-7
    return np.sqrt(np.diag(gradient @ fit.covariance @ gradient.T))


def coordinate_nuisance(q: np.ndarray) -> RateNuisances:
    """Construct public nuisances from a rate-fit vector in dominant D0 slots."""

    def probability(pair: np.ndarray) -> np.ndarray:
        s, f = pair
        return np.array([s * f, s * (1 - f), 1 - s])

    d0 = np.empty((3, 3))
    for row, slot in enumerate(((0, 1, 2), (1, 0, 2), (2, 0, 1))):
        d0[row, list(slot)] = probability(q[3 + 2 * row : 5 + 2 * row])
    return RateNuisances(
        d0,
        q[9:15].reshape(3, 2),
        float(q[15]),
        probability(q[16:18]),
        q[18:21],
        probability(q[21:23]),
    )


def oracle_coordinates() -> np.ndarray:
    return np.array(
        [
            0.05,
            0.08,
            0.025,  # rates, inverse microseconds
            0.9,
            0.8,
            0.85,
            0.9,
            0.9,
            0.8,  # D0, dominant slots 0,1,2
            -1.0,
            0.1,
            1.0,
            0.2,
            0.1,
            1.8,
            0.25,  # centers and variance
            0.9,
            0.7,
            0.95,
            0.8,
            0.9,
            0.9,
            0.7,  # thermal, retention, landing
        ]
    )


@pytest.mark.parametrize("repeated", [False, True])
def test_rate_mean_jacobian_matches_oracle_in_every_coordinate(repeated: bool) -> None:
    q = oracle_coordinates()
    if repeated:
        # G = r*(1*pi-I) has two exactly repeated nonzero eigenvalues.
        pi = coordinate_nuisance(q).thermal
        q[:3] = [0.05 * (pi[i] + pi[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    axis = np.array([0.0, 1.7, 12.0])
    names = ("rate_01", "rate_02", "rate_12")

    def moments(values: np.ndarray):
        return rate_moments(
            axis, dict(zip(names, values[:3], strict=True)), coordinate_nuisance(values)
        )

    reference = moments(q)
    for col, name in enumerate(reference.parameter_names):
        step = 1e-6
        plus, minus = q.copy(), q.copy()
        plus[col] += step
        minus[col] -= step
        oracle = (moments(plus).means - moments(minus).means) / (2 * step)
        np.testing.assert_allclose(
            reference.jacobian[:, :, col], oracle, rtol=2e-6, atol=2e-7, err_msg=name
        )


def test_rate_moments_and_populations_match_independent_expm_truth() -> None:
    curve_nuisance = STAR.nuisance(0.0, 2.1)
    nuisance = RateNuisances(
        PI_SWAP.copy(),
        curve_nuisance.centers,
        1.0,
        THERMAL.copy(),
        curve_nuisance.read_retention,
        curve_nuisance.read_landing,
    )
    parameters = {"rate_01": 1 / 20, "rate_02": 1 / 12, "rate_12": 0.0}
    sums = STAR.sums()
    moments = rate_moments(sums.axis, parameters, nuisance)
    np.testing.assert_allclose(moments.means, sums.feature_sums / 3000, atol=1e-11)
    products = np.einsum(
        "n,nij->ij",
        sums.cycle_count,
        moments.covariance + moments.means[:, :, None] * moments.means[:, None, :],
    )
    np.testing.assert_allclose(products, sums.product_sum, rtol=1e-12, atol=1e-7)
    expected = np.array(
        [THERMAL @ PI_SWAP @ expm(STAR.generator() * x) for x in sums.axis]
    )
    np.testing.assert_allclose(
        t1_model().populations(sums.axis, parameters, nuisance), expected, atol=1e-13
    )
    zero = dict.fromkeys(parameters, 0.0)
    np.testing.assert_allclose(
        t1_model().populations(sums.axis, zero, nuisance),
        np.tile(THERMAL @ PI_SWAP, (len(sums.axis), 1)),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "init", [{}, {"t1": 0.0}, {"t1": np.inf}, {"t1": np.nan}, {"t1": 20.0, "tf": 12.0}]
)
def test_rate_fit_rejects_invalid_initial_time(init: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="init"):
        fit_cycle(STAR.sums(), t1_model(), init)


@pytest.mark.parametrize(
    "axis", [np.array([-1.0]), np.array([np.inf]), np.array([]), np.ones((2, 1))]
)
def test_rate_model_rejects_invalid_delays(axis: np.ndarray) -> None:
    q = oracle_coordinates()
    with pytest.raises(ValueError, match="delay"):
        t1_model().populations(
            axis,
            dict(zip(("rate_01", "rate_02", "rate_12"), q[:3], strict=True)),
            coordinate_nuisance(q),
        )


def shot_fit(seed: int) -> CycleFit:
    sums = STAR.sums(contrast=2.1, cycles=3000, seed=seed)

    return fit_cycle(sums, t1_model(), {"t1": 20.0})


@pytest.mark.slow
@pytest.mark.parametrize("seed", [3101, 3102, 3103, 3104, 3105])
def test_simulated_shots_give_t1_within_three_wald_errors(seed: int) -> None:
    result = shot_fit(seed)

    relaxation = result.relaxation
    assert relaxation is not None
    times, amplitudes = STAR.modes()
    truth = times[int(np.argmax(amplitudes))]
    # Five fixed replicates guard the local Wald scale on shots, not precise coverage.
    assert abs(relaxation.t1 - truth) < 3 * relaxation.t1_error
