"""Readout metric contracts at the saved-moments analysis seam."""

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_array_equal
from zcu_tools.analysis.readout_chain import ReadoutSums, readout_metric


def exact_moments(
    centres: np.ndarray, retention: np.ndarray, landing: np.ndarray
) -> np.ndarray:
    """Independent finite-state expectation of the documented 113 columns.

    Noise variance is one per component. The largest same-read power is two,
    so only E[x]=c and E[x²]=c²+1 are needed at either end of a lag.
    """
    transition = np.diag(retention) + (1 - retention[:, None]) * landing
    population = landing / (1 - retention)
    population /= population.sum()

    def emission(components: tuple[int, ...]) -> np.ndarray:
        value = np.prod(centres[:, components], axis=1)
        if len(components) == 2 and components[0] == components[1]:
            value = value + 1
        return value

    def lag(k: int, left: tuple[int, ...], right: tuple[int, ...]) -> float:
        return float(
            (population * emission(left))
            @ np.linalg.matrix_power(transition, k)
            @ emission(right)
        )

    values = [
        float(population @ emission(c)) for c in ((0,), (1,), (0, 0), (0, 1), (1, 1))
    ]
    values += [lag(k, (a,), (b,)) for k in range(1, 16) for a in (0, 1) for b in (0, 1)]
    values += [
        lag(k, (a, b), (c,))
        for k in range(1, 5)
        for a, b in ((0, 0), (0, 1), (1, 1))
        for c in (0, 1)
    ]
    values += [
        lag(k, (a,), (b, c))
        for k in range(1, 5)
        for a in (0, 1)
        for b, c in ((0, 0), (0, 1), (1, 1))
    ]
    return np.array(values)


def saved_moments(features: np.ndarray, blocks: int = 1_000_000) -> ReadoutSums:
    """Wrap moment averages in the saved processor contract."""
    features = np.atleast_2d(features)
    output = np.column_stack((np.full(len(features), blocks), features * blocks))
    return ReadoutSums.from_processor(np.arange(len(features), dtype=float), output)


def test_exact_two_state_moments_recover_reference_information() -> None:
    features = exact_moments(
        np.array([[-1.5, 0.0], [1.5, 0.0]]),
        np.array([0.75, 0.75]),
        np.array([0.5, 0.5]),
    )

    result = readout_metric(saved_moments(features))

    # Asymptotic SD of the two-state estimator for this stream at 10⁶ blocks,
    # from the research two-state fit; an unseen third state must not inflate
    # the error or bias the metric beyond it.
    two_state_sd = 0.0010496
    # rdr_metric.rdr(3, .75), independent read-drive-read reference.
    assert abs(result.metric[0] - 1.4764556962025317) < two_state_sd
    assert 0.5 * two_state_sd < result.standard_error[0] < 2 * two_state_sd
    np.testing.assert_allclose(result.separation, [3], rtol=0.01)
    np.testing.assert_allclose(result.retention, [[0.75, 0.75]], atol=0.005)
    assert result.third_population[0] < 0.01


@pytest.mark.parametrize("separation", [1.0, 3.0])
def test_exact_three_state_moments_recover_dominant_pair_and_leakage(
    separation: float,
) -> None:
    centres = separation * np.array([[-0.5, 0], [0.5, 0], [0, 1]])
    features = exact_moments(
        centres, np.array([0.75, 0.75, 0.9]), np.array([0.49, 0.49, 0.02])
    )

    result = readout_metric(saved_moments(features))

    # rdr3.rdr_pair_info on the selected two centres/retentions (K=2).
    reference = {1.0: 0.09207161125319693, 3.0: 1.4764556962025317}
    np.testing.assert_allclose(result.metric, [reference[separation]], rtol=0.01)
    np.testing.assert_allclose(result.separation, [separation], rtol=0.01)
    np.testing.assert_allclose(result.retention, [[0.75, 0.75]], atol=0.005)
    # Stationary π_f = (.02/.1)/(.49/.25 + .49/.25 + .02/.1).
    np.testing.assert_allclose(
        result.third_population, [0.04854368932038836], atol=0.002
    )


def test_settings_are_ranked_by_information_not_acquisition_order() -> None:
    features = np.array(
        [
            exact_moments(
                s * np.array([[-0.5, 0], [0.5, 0], [0, 1]]),
                np.array([0.75, 0.75, 0.9]),
                np.array([0.49, 0.49, 0.02]),
            )
            for s in (2.0, 1.0, 3.0)
        ]
    )

    result = readout_metric(saved_moments(features))

    assert_array_equal(result.ranking, [2, 0, 1])
    assert np.all(result.standard_error > 0)


def test_unidentifiable_constant_stream_fails_with_a_signal_error() -> None:
    with pytest.raises(ValueError, match="signal"):
        readout_metric(saved_moments(np.zeros(113)))


@njit(cache=True)
def stationary_block_sums(seed: int, blocks: int, landing_f: float) -> np.ndarray:
    """Simulate one stationary stream; lag anchors never cross a 32-read edge."""
    np.random.seed(seed)
    centres = np.array([[-1.5, 0.0], [1.5, 0.0], [0.0, 3.0]])
    retention = np.array([0.75, 0.75, 0.9])
    landing = np.array([(1 - landing_f) / 2, (1 - landing_f) / 2, landing_f])
    stationary = landing / (1 - retention)
    stationary /= stationary.sum()
    state = int(np.searchsorted(np.cumsum(stationary), np.random.random()))
    cumulative_landing = np.cumsum(landing)
    sums = np.zeros(113)
    reads = np.empty((32, 2))
    for _ in range(blocks):
        for t in range(32):
            for component in range(2):
                reads[t, component] = centres[state, component] + np.random.randn()
            if np.random.random() >= retention[state]:
                state = int(np.searchsorted(cumulative_landing, np.random.random()))
        accumulate_second_moments(reads, sums)
        accumulate_third_moments(reads, sums)
    return sums


@njit(cache=True)
def accumulate_second_moments(reads: np.ndarray, sums: np.ndarray) -> None:
    """Add the first 65 documented block averages to test-generated sums."""
    for t in range(32):
        sums[:2] += reads[t] / 32
        for j, (a, b) in enumerate(((0, 0), (0, 1), (1, 1))):
            sums[2 + j] += reads[t, a] * reads[t, b] / 32
    column = 5
    for lag in range(1, 16):
        for a in range(2):
            for b in range(2):
                for t in range(32 - lag):
                    sums[column] += reads[t, a] * reads[t + lag, b] / (32 - lag)
                column += 1


@njit(cache=True)
def accumulate_third_moments(reads: np.ndarray, sums: np.ndarray) -> None:
    """Add both third-order feature families, using only within-block pairs."""
    pairs = ((0, 0), (0, 1), (1, 1))
    column = 65
    for lag in range(1, 5):
        for a, b in pairs:
            for c in range(2):
                for t in range(32 - lag):
                    sums[column] += (
                        reads[t, a] * reads[t, b] * reads[t + lag, c] / (32 - lag)
                    )
                column += 1
    for lag in range(1, 5):
        for a in range(2):
            for b, c in pairs:
                for t in range(32 - lag):
                    sums[column] += (
                        reads[t, a] * reads[t + lag, b] * reads[t + lag, c] / (32 - lag)
                    )
                column += 1


def test_contiguous_stream_error_covers_truth_within_statistical_scatter() -> None:
    blocks = 50_000
    estimates, errors = [], []
    for seed in (17, 29, 43, 71):
        features = stationary_block_sums(seed, blocks, 0.02) / blocks
        result = readout_metric(saved_moments(features, blocks))
        estimates.append(result.metric[0])
        errors.append(result.standard_error[0])
    z = (np.array(estimates) - 1.4764556962025317) / np.array(errors)

    assert np.all(np.isfinite(z))
    # Four fixed streams are a scatter check, not a calibrated 68% coverage study.
    assert np.max(np.abs(z)) < 3
    assert abs(np.mean(z)) < 1.5
    assert 0.2 < np.std(z, ddof=1) < 2.5


def test_processor_record_preserves_counts_axis_and_all_moments() -> None:
    axis = np.array([7.0, 3.0])
    output = np.arange(228.0).reshape(2, 114)
    output[:, 0] = [23, 91]

    record = ReadoutSums.from_processor(axis, output)

    assert_array_equal(record.axis, [7, 3])
    assert_array_equal(record.block_count, [23, 91])
    assert_array_equal(record.feature_sums, output[:, 1:])
    output[:] = 0
    axis[:] = 0
    assert_array_equal(record.axis, [7, 3])
    assert_array_equal(record.block_count, [23, 91])


@pytest.mark.parametrize(
    "axis, counts, features",
    [
        (np.empty(0), np.empty(0), np.empty((0, 113))),
        (np.ones((2, 1)), np.ones(2) * 2, np.zeros((2, 113))),
        (np.arange(2.0), np.ones(3) * 2, np.zeros((2, 113))),
        (np.arange(2.0), np.ones(2) * 2, np.zeros((2, 112))),
        (np.array([np.nan, 1]), np.ones(2) * 2, np.zeros((2, 113))),
        (np.arange(2.0), np.array([np.inf, 2]), np.zeros((2, 113))),
        (np.arange(2.0), np.array([1, 2]), np.zeros((2, 113))),
        (np.arange(2.0), np.array([2.5, 2]), np.zeros((2, 113))),
        (np.arange(2.0), np.ones(2) * 2, np.full((2, 113), np.nan)),
    ],
)
def test_invalid_saved_moments_fail_at_construction(
    axis: np.ndarray, counts: np.ndarray, features: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="shape|finite|blocks|integer"):
        ReadoutSums(axis, counts, features)


@pytest.mark.parametrize("output", [np.zeros((2, 113)), np.zeros((3, 114))])
def test_processor_output_requires_one_row_per_setting(output: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        ReadoutSums.from_processor(np.arange(2.0), output)


# Reference outputs of the one-step estimator. Cases 2-6 were recorded with the
# previous term-by-term contiguous covariance. Case 1 (no third state) was
# recorded after the start began an unobserved third mode at λ 0.9.
@pytest.mark.parametrize(
    "centres, retention, landing, expected",
    [
        (
            [-1.5, 0.0, 1.5, 0.0, 0.0, 0.0],
            [0.75, 0.75, 0.9],
            [0.5, 0.5, 0.0],
            [
                1.4763772317804855,
                0.0009352910110219197,
                2.99979842503713,
                0.750014551469412,
                0.7500183638087144,
                0.004518775269549567,
            ],
        ),
        (
            [-0.5, 0.0, 0.5, 0.0, 0.0, 1.0],
            [0.75, 0.75, 0.9],
            [0.49, 0.49, 0.02],
            [
                0.09216337861308682,
                0.00019892468738077603,
                1.0003030358841285,
                0.7499691499772169,
                0.7500340482845427,
                0.048542437188464634,
            ],
        ),
        (
            [-1.5, 0.0, 1.5, 0.0, 0.0, 3.0],
            [0.75, 0.75, 0.9],
            [0.49, 0.49, 0.02],
            [
                1.4772180779480517,
                0.0009615318058303575,
                3.000929827163965,
                0.7499968471325029,
                0.7499999266526987,
                0.048547370319327904,
            ],
        ),
        (
            [
                -1.3523439306670328,
                0.09406588960697114,
                1.523982675052278,
                -0.18764031172329215,
                0.07468568823196649,
                3.143443118712327,
            ],
            [0.4902133691881038, 0.53675991026555, 0.5678022222108959],
            [0.48, 0.5, 0.02],
            [
                0.5520919229644344,
                0.0005930573567781496,
                2.8679430777401715,
                0.5421268362982206,
                0.4929031734672349,
                0.02006797461225399,
            ],
        ),
        (
            [
                -1.3952033937401496,
                -0.2572736311776651,
                1.3664532872555633,
                -0.0008210135063994839,
                -0.07355092845686388,
                3.011773487701744,
            ],
            [0.98, 0.99, 0.995],
            [0.48, 0.5, 0.02],
            [
                2.888183658948877,
                0.0013040888935517152,
                2.7688297954224277,
                0.9877966999190795,
                0.9829655591524091,
                0.050940981298650774,
            ],
        ),
        (
            [
                -1.6227092400162915,
                -0.20366750552178575,
                1.6525731838737987,
                0.037044786507506706,
                0.09304698437955058,
                2.923058357019677,
            ],
            [0.999, 0.9995, 0.9999],
            [0.48, 0.5, 0.02],
            [
                3.892569328007108,
                0.0014556867459719906,
                3.1084794162571203,
                0.9917556844268278,
                0.9942363591926585,
                0.08178195652262832,
            ],
        ),
    ],
)
def test_contiguous_covariance_preserves_saved_moment_inference(
    centres: list[float],
    retention: list[float],
    landing: list[float],
    expected: list[float],
) -> None:
    """Retire these references only when the one-step estimator contract changes."""
    features = exact_moments(
        np.array(centres).reshape(-1, 2), np.array(retention), np.array(landing)
    )

    result = readout_metric(saved_moments(features))

    actual = np.concatenate(
        (
            result.metric,
            result.standard_error,
            result.separation,
            result.retention.ravel(),
            result.third_population,
        )
    )
    # The error's finite-difference gradient amplifies covariance rounding to 1e-8.
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0)
