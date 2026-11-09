"""Readout metric contracts at the saved-moments analysis seam."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_array_equal
from scipy.optimize import OptimizeResult, least_squares
from zcu_tools.analysis.readout_chain import ReadoutSums, _estimator, readout_metric


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

    fit = readout_metric(saved_moments(features)).fits[0]

    # Independent truncated-lag GLS at q=2.25, Q=.75. A three-state fit may
    # assign a small unseen third population, so allow its modelling error.
    np.testing.assert_allclose(fit.metric, 2.3030409955595212, rtol=0.02)
    assert 0 < fit.standard_error < 0.01
    np.testing.assert_allclose(fit.separation, 3, rtol=0.01)
    np.testing.assert_allclose(fit.retention, [0.75, 0.75], atol=0.005)
    assert fit.third_population < 0.01


@pytest.mark.parametrize("separation", [1.0, 3.0])
def test_exact_three_state_moments_recover_dominant_pair_and_leakage(
    separation: float,
) -> None:
    centres = separation * np.array([[-0.5, 0], [0.5, 0], [0, 1]])
    features = exact_moments(
        centres, np.array([0.75, 0.75, 0.9]), np.array([0.49, 0.49, 0.02])
    )

    fit = readout_metric(saved_moments(features)).fits[0]

    # Independent truncated-lag GLS, Q=.75/(1-.25*.02) after row normalisation.
    reference = {1.0: 0.5430849373534309, 3.0: 2.364008370477314}
    np.testing.assert_allclose(fit.metric, reference[separation], rtol=0.01)
    np.testing.assert_allclose(fit.separation, separation, rtol=0.01)
    np.testing.assert_allclose(fit.retention, [0.75, 0.75], atol=0.005)
    np.testing.assert_allclose(fit.qnd, 0.75 / 0.995, rtol=0.002)
    assert 0 < fit.qnd_error < 0.01
    # Stationary π_f = (.02/.1)/(.49/.25 + .49/.25 + .02/.1).
    np.testing.assert_allclose(fit.third_population, 0.04854368932038836, atol=0.002)


@pytest.mark.parametrize(
    "retention, reference, qnd",
    [
        ([0.7, 0.8, 0.9], 2.3631887130074527, 0.7537191017591494),
        ([0.5, 0.5, 0.95], 0.5899480231069921, 0.505050505050505),
    ],
)
def test_bounded_refinement_recovers_unequal_and_low_retention_truth(
    retention: list[float], reference: float, qnd: float
) -> None:
    features = exact_moments(
        3 * np.array([[-0.5, 0], [0.5, 0], [0, 1]]),
        np.array(retention),
        np.array([0.49, 0.49, 0.02]),
    )

    fit = readout_metric(saved_moments(features)).fits[0]

    # References are truncated-lag GLS at q=2.25 and the truth's normalised Q.
    np.testing.assert_allclose(fit.metric, reference, rtol=0.002)
    np.testing.assert_allclose(fit.separation, 3, rtol=0.002)
    np.testing.assert_allclose(sorted(fit.retention), sorted(retention[:2]), atol=0.002)
    np.testing.assert_allclose(fit.qnd, qnd, rtol=0.002)


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

    assert result.ranking == (2, 0, 1)
    assert all(fit.standard_error > 0 for fit in result.fits.values())


def test_setting_without_signal_is_reported_apart_and_is_not_ranked() -> None:
    signal = exact_moments(
        3 * np.array([[-0.5, 0], [0.5, 0], [0, 1]]),
        np.array([0.75, 0.75, 0.9]),
        np.array([0.49, 0.49, 0.02]),
    )
    features = np.array([signal, np.zeros(113), signal])
    sums = ReadoutSums(
        axis=np.array([5.0, 6.0, 7.0]),
        block_count=np.full(3, 1e6),
        feature_sums=features * 1e6,
    )

    result = readout_metric(sums)

    assert list(result.fits) == [0, 2]
    assert result.no_signal == (1,)
    assert result.failures == {}
    assert result.fits[2].setting == 7.0
    np.testing.assert_allclose(result.fits[2].metric, 2.364008370477314, rtol=0.01)
    assert result.ranking == (0, 2)


@pytest.mark.parametrize(
    "separation, expected_signal", [(0.0, False), (0.03, False), (0.3, True)]
)
def test_no_signal_test_separates_noise_from_resolvable_contrast(
    separation: float, expected_signal: bool
) -> None:
    features = exact_moments(
        np.array([2.0, -1.0]) + separation * np.array([[-0.5, 0], [0.5, 0], [0, 1]]),
        np.array([0.75, 0.75, 0.9]),
        np.array([0.49, 0.49, 0.02]),
    )

    result = readout_metric(saved_moments(features, blocks=200_000))

    assert (0 in result.fits) == expected_signal
    assert result.no_signal == (() if expected_signal else (0,))
    assert result.failures == {}
    assert result.ranking == ((0,) if expected_signal else ())


def test_no_fitted_setting_returns_no_signal_and_an_empty_ranking() -> None:
    result = readout_metric(saved_moments(np.zeros(113)))

    assert result.fits == {}
    assert result.no_signal == (0,)
    assert result.failures == {}
    assert result.ranking == ()


def test_evaluation_budget_failure_does_not_discard_other_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def limited_solve(
        fun: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        *,
        jac: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[np.ndarray, np.ndarray],
        method: str,
        x_scale: str,
        max_nfev: int,
    ) -> OptimizeResult:
        nonlocal calls
        calls += 1
        # Delegate to the real external optimizer, but exhaust the first budget.
        # scipy-stubs restricts jac to str; SciPy accepts a callable.
        analytic: Any = jac
        return least_squares(
            fun,
            x0,
            jac=analytic,
            bounds=bounds,
            method=method,
            x_scale=x_scale,
            max_nfev=1 if calls == 1 else max_nfev,
        )

    monkeypatch.setattr(_estimator, "least_squares", limited_solve)
    features = exact_moments(
        3 * np.array([[-0.5, 0], [0.5, 0], [0, 1]]),
        np.array([0.75, 0.75, 0.9]),
        np.array([0.49, 0.49, 0.02]),
    )

    result = readout_metric(saved_moments(np.array([features, features])))

    assert list(result.failures) == [0]
    assert "evaluation budget" in result.failures[0].reason
    assert result.failures[0].setting == 0.0
    assert list(result.fits) == [1]
    assert result.no_signal == ()
    assert result.ranking == (1,)


def test_record_changed_after_construction_still_fails_the_contract() -> None:
    sums = saved_moments(np.zeros((2, 113)))
    # The frozen record's arrays stay writable after construction.
    sums.feature_sums[1, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        readout_metric(sums)


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
        fit = readout_metric(saved_moments(features, blocks)).fits[0]
        estimates.append(fit.metric)
        errors.append(fit.standard_error)
    z = (np.array(estimates) - 2.364008370477314) / np.array(errors)

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
