"""Thermal-chain analysis through saved sums and heuristic score contracts."""

import numpy as np
import pytest
from zcu_tools.analysis.readout_chain import (
    ThermalChainSums,
    ThermalMomentsUnavailable,
    thermal_chain_score,
)


def test_saved_sums_copy_processor_output_and_preserve_additive_columns():
    axis = np.array([2.0, 1.0])
    columns = np.arange(216, dtype=float).reshape(2, 108)
    columns[:, 0] = [1, 640_000]
    saved = ThermalChainSums.from_processor(axis, columns)
    np.testing.assert_array_equal(saved.axis, axis)
    np.testing.assert_array_equal(saved.shot_count, columns[:, 0])
    np.testing.assert_array_equal(saved.feature_sums, columns[:, 1:])
    columns[:] = -1
    axis[:] = -1
    np.testing.assert_array_equal(saved.axis, [2, 1])
    np.testing.assert_array_equal(saved.shot_count, [1, 640_000])
    assert saved.feature_sums[1, -1] == 215


@pytest.mark.parametrize(
    ("axis", "count", "features"),
    [
        (np.zeros(0), np.ones(0), np.ones((0, 107))),
        (np.zeros((1, 1)), np.ones(1), np.ones((1, 107))),
        (np.zeros(2), np.ones(1), np.ones((2, 107))),
        (np.zeros(1), np.ones((1, 1)), np.ones((1, 107))),
        (np.zeros(1), np.ones(1), np.ones((1, 106))),
        (np.zeros(1), np.ones(1), np.ones((1, 108))),
        (np.zeros(1), np.zeros(1), np.ones((1, 107))),
        (np.zeros(1), np.array([-1.0]), np.ones((1, 107))),
        (np.zeros(1), np.array([1.5]), np.ones((1, 107))),
        (np.zeros(1), np.array([np.inf]), np.ones((1, 107))),
        (np.array([np.nan]), np.ones(1), np.ones((1, 107))),
        (np.zeros(1), np.ones(1), np.full((1, 107), np.inf)),
        (np.zeros(1), np.ones(1), np.full((1, 107), np.nan)),
    ],
)
def test_saved_sums_reject_malformed_or_nonfinite_arrays(axis, count, features):
    with pytest.raises(ValueError, match="expected.*actual"):
        ThermalChainSums(axis, count, features)


@pytest.mark.parametrize("shape", [(1, 107), (1, 109), (2, 108), (108,)])
def test_processor_admission_rejects_wrong_column_or_setting_counts(shape):
    with pytest.raises(ValueError, match="expected.*108.*actual"):
        ThermalChainSums.from_processor(np.array([0.0]), np.ones(shape))


def chain_moments(centres, transition, start, noise=1.0):
    """Finite-state oracle: matrix powers and independent Gaussian emissions."""
    populations = [start @ np.linalg.matrix_power(transition, r) for r in range(16)]
    means = [population @ centres for population in populations]
    same = centres.T @ np.diag(start) @ centres + noise**2 * np.eye(2)
    pairs = []
    for a, b in [*((r, r + 1) for r in range(15)), (0, 2), (0, 3), (0, 4)]:
        joint = np.diag(populations[a]) @ np.linalg.matrix_power(transition, b - a)
        pairs.append(centres.T @ joint @ centres)
    return np.concatenate(
        [np.ravel(means), same[[0, 0, 1], [0, 1, 1]], np.ravel(pairs)]
    )


def saved_moments(features, shots=640_000):
    features = np.atleast_2d(features)
    return ThermalChainSums.from_processor(
        np.arange(len(features), dtype=float),
        np.column_stack([np.full(len(features), shots), features * shots]),
    )


def amplification_oracle(rho: float, qnd: float) -> float:
    """Literal lag-by-lag information calculation, independent of production."""
    eta = np.clip(qnd, 0.001, 0.995)
    s = min(abs(rho) / eta, 0.999)
    q = s / (1 - s)
    nodes, weights = np.polynomial.legendre.leggauss(24)

    def information(determinant):
        u = eta * determinant
        if q == 0:
            return sum(j**2 * u ** (2 * (j - 1)) for j in range(1, 16))
        b = (1 + u**2) / (1 - u**2)
        jvector = np.array([j * q * eta * u ** (j - 1) for j in range(1, 16)])
        covariance = np.empty((15, 15))
        for j in range(1, 16):
            for ell in range(1, 16):
                d, total = abs(j - ell), j + ell
                covariance[j - 1, ell - 1] = (
                    (j == ell)
                    + 2 * q * (u**d + u**total)
                    + q**2 * ((d + b) * u**d - (total + b) * u**total)
                )
        return jvector @ np.linalg.solve(covariance, jvector)

    integral = sum(
        w * information((x + 1) / 2) / 2 for x, w in zip(nodes, weights, strict=True)
    )
    return float(integral / information(0))


@pytest.mark.parametrize(
    ("population", "separation", "qnd"),
    [(0.5, 3.0, 0.75), (0.1, 0.3, 0.95), (0.7, 1.2, 0.3)],
)
def test_stationary_two_state_score_matches_closed_forms(population, separation, qnd):
    start = np.array([1 - population, population])
    transition = qnd * np.eye(2) + (1 - qnd) * np.tile(start, (2, 1))
    centres = np.outer([-separation / 2, separation / 2], [0.6, 0.8])
    result = thermal_chain_score(
        saved_moments(chain_moments(centres, transition, start))
    )
    fit = result.settings[0]
    signal_variance = population * (1 - population) * separation**2
    rho = signal_variance * qnd / (1 + signal_variance)
    np.testing.assert_allclose(fit.correlation, rho, rtol=1e-9)
    np.testing.assert_allclose(
        fit.score, rho**2 * amplification_oracle(rho, qnd), rtol=1e-9
    )
    np.testing.assert_allclose(fit.qnd, qnd, rtol=1e-9)
    np.testing.assert_allclose(fit.mist, 0, atol=1e-9)
    assert fit.setting == 0
    assert result.ranking == (0,)


def three_state_model():
    return (
        np.array([[-0.6, 0.2], [0.8, -0.1], [2.0, 3.0]]),
        np.array([[0.88, 0.09, 0.03], [0.15, 0.70, 0.15], [0.03, 0.07, 0.90]]),
        np.array([0.6, 0.4, 0.0]),
    )


def reference_components(centres, transition, start, noise=1.0):
    """Direct population-centred covariance, without saved-column decoding."""

    def covariance(a, b):
        population = start @ np.linalg.matrix_power(transition, a)
        later = population @ np.linalg.matrix_power(transition, b - a)
        joint = np.diag(population) @ np.linalg.matrix_power(transition, b - a)
        return centres.T @ (joint - np.outer(population, later)) @ centres

    covariances = np.array([covariance(0, lag) for lag in range(1, 5)])
    covariances = (covariances + covariances.transpose(0, 2, 1)) / 2
    axis = np.linalg.eigh(covariances.sum(axis=0))[1][:, -1]
    variance = covariance(0, 0) + noise**2 * np.eye(2)
    v = axis @ variance @ axis
    lag = np.array([axis @ covariance @ axis for covariance in covariances])
    rho = lag[0] / v
    shift = (np.eye(4, k=1) + np.eye(4, k=-1)) / 2
    head = np.diag([1, 1, 1, 0])
    qnd = (lag @ shift @ lag) / (lag @ head @ lag)
    gradient = 2 * (shift @ lag - qnd * head @ lag) / (lag @ head @ lag)
    difference = np.zeros((2, 2))
    for r in range(1, 16):
        difference += (r - 8) / 20 * covariance(r - 1, r)
    mist = np.linalg.eigvalsh((difference + difference.T) / 2)[-1] / v
    return float(rho), float(qnd), float(v * np.linalg.norm(gradient)), float(mist)


def heuristic_oracle(rho, qnd, mist):
    if rho == 0:
        return 0.0
    return (
        rho**2
        * amplification_oracle(rho, qnd)
        * (abs(rho) / (abs(rho) + max(mist, 0))) ** 2
    )


def projected_moments(lags, mist=0.0):
    """Saved zero-mean IQ moments with known x-axis projections and v=1."""
    moments = np.zeros(107)
    moments[32:35] = [1, 0, 0.5]
    for r in range(15):
        moments[35 + 4 * r] = lags[0] + r * mist / 14
    moments[95::4] = lags[1:]
    return moments


@pytest.mark.parametrize(
    "rho,qnd,mist", [(0.2, 0.7, 0), (0.3, 0.8, 0.2), (-0.1, 0.12, 0.1)]
)
def test_model_error_scale_matches_finite_difference_only_inside_clip_domain(
    rho, qnd, mist
):
    lags = np.array([rho * qnd**j for j in range(4)])
    if rho < 0:
        lags = np.array([-0.1, 0.15, 0.1, 0.05])
    shots = 900_000
    fit = thermal_chain_score(
        saved_moments(projected_moments(lags, mist), shots)
    ).settings[0]
    step = 1e-5
    drho = (
        heuristic_oracle(fit.correlation + step, fit.qnd, fit.mist)
        - heuristic_oracle(fit.correlation - step, fit.qnd, fit.mist)
    ) / (2 * step)
    dqnd = (
        heuristic_oracle(fit.correlation, fit.qnd + step, fit.mist)
        - heuristic_oracle(fit.correlation, fit.qnd - step, fit.mist)
    ) / (2 * step)
    expected = np.hypot(
        drho * np.sqrt((1 + fit.correlation**2) / shots), dqnd * fit.qnd_error
    )
    np.testing.assert_allclose(fit.score_error, expected, rtol=1e-6)


def amplification_slopes_oracle(rho, qnd):
    """Differentiate literal J,V information in independent eta,s coordinates."""
    eta = np.clip(qnd, 0.001, 0.995)
    s = min(abs(rho) / eta, 0.999)
    q, qs = s / (1 - s), 1 / (1 - s) ** 2
    nodes, weights = np.polynomial.legendre.leggauss(24)
    integral = np.zeros(3)
    for node, weight in zip(nodes, weights, strict=True):
        lam = (node + 1) / 2
        u = eta * lam
        b, bu = (1 + u**2) / (1 - u**2), 4 * u / (1 - u**2) ** 2
        signal = np.array([j * q * eta * u ** (j - 1) for j in range(1, 16)])
        signal_eta = signal * np.arange(1, 16) / eta
        signal_s = signal * qs / q
        covariance, derivative_eta, derivative_s = np.zeros((3, 15, 15))
        for j in range(1, 16):
            for ell in range(1, 16):
                d, total = abs(j - ell), j + ell
                a, z = u**d, u**total
                au, zu = d * u ** (d - 1), total * u ** (total - 1)
                contrast = (d + b) * a - (total + b) * z
                covariance[j - 1, ell - 1] = (
                    (j == ell) + 2 * q * (a + z) + q**2 * contrast
                )
                derivative_eta[j - 1, ell - 1] = lam * (
                    2 * q * (au + zu)
                    + q**2 * ((d + b) * au - (total + b) * zu + bu * (a - z))
                )
                derivative_s[j - 1, ell - 1] = qs * (2 * (a + z) + 2 * q * contrast)
        solved = np.linalg.solve(covariance, signal)
        integral += (
            weight
            / 2
            * np.array(
                [
                    signal @ solved,
                    2 * signal_eta @ solved - solved @ derivative_eta @ solved,
                    2 * signal_s @ solved - solved @ derivative_s @ solved,
                ]
            )
        )
    normalization = (s * eta) ** 2
    gain = integral[0] / normalization
    gain_eta = integral[1] / normalization - 2 * gain / eta
    gain_s = integral[2] / normalization - 2 * gain / s
    return gain, np.sign(rho) * gain_s / eta, gain_eta - s * gain_s / eta


@pytest.mark.parametrize(
    "rho,qnd",
    [
        (0.0002, 0.001),
        (0.0002, -0.2),
        (0.4, 0.995),
        (0.4, 1.2),
        (0.6993, 0.7),
        (0.9, 0.7),
        (0.2, -0.1),
    ],
)
def test_clipped_error_uses_model_slopes_not_clipped_output_derivatives(rho, qnd):
    lags = rho * qnd ** np.arange(4)
    moments = projected_moments(lags, mist=0.2)
    shots = 800_000
    fit = thermal_chain_score(saved_moments(moments, shots)).settings[0]
    gain, gain_rho, gain_qnd = amplification_slopes_oracle(fit.correlation, fit.qnd)
    prefactor = rho**4 / (abs(rho) + fit.mist) ** 2
    derivative = np.sign(rho) * (
        4 * abs(rho) ** 3 / (abs(rho) + fit.mist) ** 2
        - 2 * rho**4 / (abs(rho) + fit.mist) ** 3
    )
    expected = np.hypot(
        (derivative * gain + prefactor * gain_rho) * np.sqrt((1 + rho**2) / shots),
        prefactor * gain_qnd * fit.qnd_error,
    )
    np.testing.assert_allclose(fit.score_error, expected, rtol=1e-6)
    np.testing.assert_allclose(
        fit.score, heuristic_oracle(rho, qnd, fit.mist), rtol=1e-9
    )
    assert np.all(np.isfinite(score_values(fit)))


@pytest.mark.parametrize("q", [1e-4, 0.01, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("qnd", [0.01, 0.3, 0.8, 0.995])
def test_public_amplification_matches_lag_by_lag_oracle(q, qnd):
    rho = qnd * q / (1 + q)
    fit = thermal_chain_score(
        saved_moments(projected_moments(rho * qnd ** np.arange(4)))
    ).settings[0]
    np.testing.assert_allclose(
        fit.amplification, amplification_oracle(rho, qnd), rtol=1e-9
    )


def test_zero_correlation_with_estimable_qnd_uses_continuous_score_and_error_limit():
    fit = thermal_chain_score(
        saved_moments(projected_moments([0, 0.1, 0.04, 0.01]))
    ).settings[0]
    assert fit.correlation == fit.score == fit.score_error == 0
    np.testing.assert_allclose(
        fit.amplification, amplification_oracle(0, fit.qnd), rtol=1e-9
    )


def test_ranking_uses_only_score_and_preserves_exact_ties_without_exclusion():
    moments = [
        projected_moments(0.3 * 0.8 ** np.arange(4), mist=m) for m in [0, 2, 0, 0.1]
    ]
    counts = np.array([2, 2**20, 2**20, 2**20])
    saved = ThermalChainSums(
        np.array([8.0, 2.0, 4.0, 1.0]), counts, np.array(moments) * counts[:, None]
    )
    result = thermal_chain_score(saved)
    assert result.settings[1].mist > abs(result.settings[1].correlation)
    assert result.settings[0].score == result.settings[2].score
    assert result.settings[0].score_error > result.settings[2].score_error
    assert result.ranking == (0, 2, 3, 1)
    assert [fit.setting for fit in result.settings] == [8, 2, 4, 1]


def score_values(setting):
    return np.array(
        [
            setting.score,
            setting.score_error,
            setting.correlation,
            setting.amplification,
            setting.qnd,
            setting.qnd_error,
            setting.mist,
        ]
    )


@pytest.mark.parametrize("noise", [0.4, 1.0, 2.0])
@pytest.mark.parametrize("states", [2, 3])
def test_nonstationary_outputs_match_independent_matrix_power_oracle(noise, states):
    centres, transition, start = three_state_model()
    if states == 2:
        centres = centres[:2]
        equilibrium = np.array([0.2, 0.8])
        transition = 0.8 * np.eye(2) + 0.2 * np.tile(equilibrium, (2, 1))
        start = np.array([0.95, 0.05])
    shots = 200_000
    saved = saved_moments(chain_moments(centres, transition, start, noise), shots)
    fit = thermal_chain_score(saved).settings[0]
    rho, qnd, qnd_scale, mist = reference_components(centres, transition, start, noise)
    assert mist > 0
    np.testing.assert_allclose(
        [
            fit.score,
            fit.correlation,
            fit.amplification,
            fit.qnd,
            fit.qnd_error,
            fit.mist,
        ],
        [
            heuristic_oracle(rho, qnd, mist),
            rho,
            amplification_oracle(rho, qnd),
            qnd,
            qnd_scale / np.sqrt(shots),
            mist,
        ],
        rtol=1e-9,
    )


def test_constant_iq_offset_leaves_every_score_output_unchanged():
    centres, transition, start = three_state_model()
    baseline = thermal_chain_score(
        saved_moments(chain_moments(centres, transition, start))
    )
    shifted = thermal_chain_score(
        saved_moments(chain_moments(centres + [7, -13], transition, start))
    )
    np.testing.assert_allclose(
        score_values(shifted.settings[0]), score_values(baseline.settings[0]), rtol=1e-9
    )
    assert shifted.settings[0].setting == baseline.settings[0].setting
    assert shifted.ranking == baseline.ranking


@pytest.mark.parametrize("constant", [[0.0, 0.0], [3.0, -2.0]])
def test_constant_shots_fail_at_score_admission(constant):
    moments = chain_moments(np.array([constant]), np.ones((1, 1)), np.ones(1), noise=0)
    with pytest.raises(ValueError, match="expected variance.*actual"):
        thermal_chain_score(saved_moments(moments))


def test_zero_first_read_covariances_fail_even_with_positive_noise_variance():
    moments = chain_moments(np.zeros((1, 2)), np.ones((1, 1)), np.ones(1))
    with pytest.raises(ValueError, match="covariance norm.*actual"):
        thermal_chain_score(saved_moments(moments))


@pytest.mark.parametrize(
    "case", ["single-shot-roundoff", "constant", "zero-covariance"]
)
def test_score_reports_unestimable_moments_separately_from_malformed_sums(case):
    if case == "single-shot-roundoff":
        # Positive centring residue must not make one shot appear estimable.
        moments = projected_moments([1e-16, 1e-16, 1e-16, 1e-16])
        moments[32] = 1e-16
        sums = saved_moments(moments, shots=1)
    elif case == "constant":
        sums = saved_moments(np.zeros(107))
    else:
        sums = saved_moments(projected_moments([0, 0, 0, 0]))

    with pytest.raises(ThermalMomentsUnavailable, match="setting"):
        thermal_chain_score(sums)


def test_score_revalidates_borrowed_saved_arrays():
    centres, transition, start = three_state_model()
    saved = saved_moments(chain_moments(centres, transition, start))
    saved.feature_sums[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite.*actual non-finite") as invalid:
        thermal_chain_score(saved)
    assert not isinstance(invalid.value, ThermalMomentsUnavailable)


@pytest.mark.parametrize("field", ["axis", "shot_count", "feature_sums"])
@pytest.mark.parametrize("dtype", [str, complex, bool])
def test_saved_sums_reject_nonreal_or_nonnumeric_moments(field, dtype):
    fields = {
        "axis": np.array([0.0]),
        "shot_count": np.array([100.0]),
        "feature_sums": np.ones((1, 107)),
    }
    fields[field] = fields[field].astype(dtype)
    with pytest.raises(ValueError, match="real numeric"):
        ThermalChainSums(**fields)
