"""Thermal-chain analysis through saved sums and closed-form score contracts."""

import numpy as np
import pytest
from zcu_tools.analysis.readout_chain import ThermalChainSums, thermal_chain_score


def test_saved_sums_copy_processor_output_and_preserve_additive_columns():
    axis = np.array([2.0, 1.0])
    columns = np.arange(76, dtype=float).reshape(2, 38)
    columns[:, 0] = [1, 640_000]
    saved = ThermalChainSums.from_processor(axis, columns)

    np.testing.assert_array_equal(saved.axis, axis)
    np.testing.assert_array_equal(saved.shot_count, columns[:, 0])
    np.testing.assert_array_equal(saved.feature_sums, columns[:, 1:])
    columns[:] = -1
    axis[:] = -1
    np.testing.assert_array_equal(saved.axis, [2, 1])
    np.testing.assert_array_equal(saved.shot_count, [1, 640_000])
    assert saved.feature_sums[1, -1] == 75


@pytest.mark.parametrize(
    ("axis", "count", "features"),
    [
        (np.zeros(0), np.ones(0), np.ones((0, 37))),
        (np.zeros((1, 1)), np.ones(1), np.ones((1, 37))),
        (np.zeros(2), np.ones(1), np.ones((2, 37))),
        (np.zeros(1), np.ones((1, 1)), np.ones((1, 37))),
        (np.zeros(1), np.ones(1), np.ones((1, 36))),
        (np.zeros(1), np.ones(1), np.ones((1, 38))),
        (np.zeros(1), np.zeros(1), np.ones((1, 37))),
        (np.zeros(1), np.array([-1.0]), np.ones((1, 37))),
        (np.zeros(1), np.array([1.5]), np.ones((1, 37))),
        (np.zeros(1), np.array([np.inf]), np.ones((1, 37))),
        (np.array([np.nan]), np.ones(1), np.ones((1, 37))),
        (np.zeros(1), np.ones(1), np.full((1, 37), np.inf)),
        (np.zeros(1), np.ones(1), np.full((1, 37), np.nan)),
    ],
)
def test_saved_sums_reject_malformed_or_nonfinite_arrays(axis, count, features):
    with pytest.raises(ValueError, match="expected.*actual"):
        ThermalChainSums(axis, count, features)


@pytest.mark.parametrize("shape", [(1, 37), (1, 39), (2, 38), (38,)])
def test_processor_admission_rejects_wrong_column_or_setting_counts(shape):
    with pytest.raises(ValueError, match="expected.*38.*actual"):
        ThermalChainSums.from_processor(np.array([0.0]), np.ones(shape))


def chain_moments(centres, transition, start, noise=1.0):
    """Finite-state oracle: matrix powers and independent Gaussian emissions."""
    positions = (0, 1, 2, 3, 4, 14, 15)
    populations = [start @ np.linalg.matrix_power(transition, r) for r in positions]
    means = [population @ centres for population in populations]
    same = centres.T @ np.diag(start) @ centres + noise**2 * np.eye(2)
    pairs = []
    for a, b in ((0, 1), (0, 2), (0, 3), (0, 4), (14, 15)):
        population = start @ np.linalg.matrix_power(transition, a)
        joint = np.diag(population) @ np.linalg.matrix_power(transition, b - a)
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


@pytest.mark.parametrize(
    ("population", "separation", "qnd"),
    [(0.5, 3.0, 0.75), (0.1, 0.3, 0.95), (0.7, 1.2, 0.3)],
)
def test_stationary_two_state_score_matches_closed_forms(population, separation, qnd):
    start = np.array([1 - population, population])
    transition = qnd * np.eye(2) + (1 - qnd) * np.tile(start, (2, 1))
    direction = np.array([0.6, 0.8])
    centres = np.outer([-separation / 2, separation / 2], direction)
    moments = chain_moments(centres, transition, start)
    shots = 640_000

    result = thermal_chain_score(saved_moments(moments, shots))

    fit = result.settings[0]
    signal_variance = population * (1 - population) * separation**2
    rho = signal_variance * qnd / (1 + signal_variance)
    np.testing.assert_allclose(fit.correlation, rho, rtol=1e-9)
    np.testing.assert_allclose(fit.score, 4 * rho**2, rtol=1e-9)
    np.testing.assert_allclose(
        fit.score_error, 8 * abs(rho) * np.sqrt((1 + rho**2) / shots), rtol=1e-9
    )
    np.testing.assert_allclose(fit.qnd, qnd, rtol=1e-9)
    np.testing.assert_allclose(fit.mist, 0, atol=1e-9)
    assert fit.setting == 0
    assert not fit.excluded
    assert result.ranking == (0,)


def three_state_model():
    return (
        np.array([[-0.6, 0.2], [0.8, -0.1], [2.0, 3.0]]),
        np.array([[0.88, 0.09, 0.03], [0.15, 0.70, 0.15], [0.03, 0.07, 0.90]]),
        np.array([0.6, 0.4, 0.0]),
    )


def reference_score(centres, transition, start, shots, noise=1.0):
    """Independent covariance oracle, including an analytic quadratic-form gradient."""

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
    difference = covariance(14, 15) - covariance(0, 1)
    mist = np.linalg.eigvalsh((difference + difference.T) / 2)[-1] / v
    return np.array(
        [
            4 * rho**2,
            8 * abs(rho) * np.sqrt((1 + rho**2) / shots),
            rho,
            qnd,
            v * np.linalg.norm(gradient) / np.sqrt(shots),
            mist,
        ]
    )


def score_values(setting):
    return np.array(
        [
            setting.score,
            setting.score_error,
            setting.correlation,
            setting.qnd,
            setting.qnd_error,
            setting.mist,
        ]
    )


@pytest.mark.parametrize("noise", [0.4, 1.0, 2.0])
def test_three_state_outputs_match_independent_matrix_power_oracle(noise):
    centres, transition, start = three_state_model()
    shots = 200_000
    saved = saved_moments(chain_moments(centres, transition, start, noise), shots)

    result = thermal_chain_score(saved)

    expected = reference_score(centres, transition, start, shots, noise)
    np.testing.assert_allclose(score_values(result.settings[0]), expected, rtol=1e-9)
    excluded = bool(expected[-1] > abs(expected[2]))
    assert result.settings[0].excluded == excluded
    assert result.ranking == (() if excluded else (0,))


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
    assert shifted.settings[0].excluded == baseline.settings[0].excluded
    assert shifted.ranking == baseline.ranking


def test_ranking_excludes_mist_and_keeps_acquisition_order_for_exact_ties():
    centres, transition, start = three_state_model()
    excluded = chain_moments(centres, transition, start)
    initial = np.array([0.6, 0.4])
    transition_ge = 0.75 * np.eye(2) + 0.25 * np.tile(initial, (2, 1))
    weak = chain_moments(np.array([[-0.2, 0.0], [0.2, 0.0]]), transition_ge, initial)
    strong = chain_moments(np.array([[-1.0, 0.0], [1.0, 0.0]]), transition_ge, initial)
    saved = saved_moments([weak, excluded, strong, strong])
    saved.axis[:] = [8, 2, 4, 1]

    result = thermal_chain_score(saved)

    assert result.ranking == (2, 3, 0)
    assert result.settings[1].excluded
    assert [setting.setting for setting in result.settings] == [8, 2, 4, 1]
    assert thermal_chain_score(saved_moments([excluded, excluded])).ranking == ()


@pytest.mark.parametrize("constant", [[0.0, 0.0], [3.0, -2.0]])
def test_constant_shots_fail_at_score_admission(constant):
    moments = chain_moments(np.array([constant]), np.ones((1, 1)), np.ones(1), noise=0)
    with pytest.raises(ValueError, match="expected variance.*actual"):
        thermal_chain_score(saved_moments(moments))


def test_zero_first_read_covariances_fail_even_with_positive_noise_variance():
    moments = chain_moments(np.zeros((1, 2)), np.ones((1, 1)), np.ones(1))
    with pytest.raises(ValueError, match="covariance norm.*actual"):
        thermal_chain_score(saved_moments(moments))


def test_score_revalidates_borrowed_saved_arrays():
    centres, transition, start = three_state_model()
    saved = saved_moments(chain_moments(centres, transition, start))
    saved.feature_sums[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite.*actual non-finite"):
        thermal_chain_score(saved)


@pytest.mark.parametrize("field", ["axis", "shot_count", "feature_sums"])
@pytest.mark.parametrize("dtype", [str, complex, bool])
def test_saved_sums_reject_nonreal_or_nonnumeric_moments(field, dtype):
    fields = {
        "axis": np.array([0.0]),
        "shot_count": np.array([100.0]),
        "feature_sums": np.ones((1, 37)),
    }
    fields[field] = fields[field].astype(dtype)
    with pytest.raises(ValueError, match="real numeric"):
        ThermalChainSums(**fields)


def test_ranking_uses_score_minus_error_not_score_alone():
    initial = np.array([0.5, 0.5])
    transition = 0.75 * np.eye(2) + 0.25 * np.tile(initial, (2, 1))
    strong = chain_moments(np.array([[-1.0, 0.0], [1.0, 0.0]]), transition, initial)
    weak = chain_moments(np.array([[-0.5, 0.0], [0.5, 0.0]]), transition, initial)
    saved = ThermalChainSums(
        np.array([3.0, 7.0]),
        np.array([1.0, 1_000_000.0]),
        np.array([strong, weak * 1_000_000]),
    )
    result = thermal_chain_score(saved)
    assert result.settings[0].score > result.settings[1].score
    assert result.ranking == (1, 0)


def test_mist_equal_to_absolute_correlation_is_not_excluded():
    initial = np.array([0.5, 0.5])
    transition = 0.5 * np.eye(2) + 0.5 * np.tile(initial, (2, 1))
    moments = chain_moments(np.array([[-1.0, 0.0], [1.0, 0.0]]), transition, initial)
    moments[-4:] = moments[17:21] + np.array([0.5, 0, 0, 0.5])
    result = thermal_chain_score(saved_moments(moments))
    assert result.settings[0].mist == abs(result.settings[0].correlation)
    assert not result.settings[0].excluded
    assert result.ranking == (0,)
