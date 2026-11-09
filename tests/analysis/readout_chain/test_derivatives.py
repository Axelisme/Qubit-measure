"""Analytic derivatives at the numerical model seams, with test-only oracles."""

from collections.abc import Callable
from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.readout_chain import _estimator, _information
from zcu_tools.analysis.readout_chain._model import (
    ModelParameters,
    contiguous_covariance,
    moment_means,
    stationary_population,
)

FloatArray = NDArray[np.float64]
LongArray = NDArray[np.longdouble]


def moment_oracle(free: FloatArray) -> LongArray:
    """Independent finite-state moments in extended precision for tiny tangents.

    Matrix contractions follow the documented column families directly. Extra
    precision keeps differencing O(1) moments useful when a floored landing
    leaves coordinate sensitivities many orders of magnitude smaller.
    """
    coordinates = free.astype(np.longdouble)
    centres = coordinates[:6].reshape(3, 2)
    variance = np.exp(coordinates[6])
    retention = np.clip(1 / (1 + np.exp(-coordinates[7:10])), 0.01, 0.995)
    weights = np.exp(np.clip(np.array([0, *coordinates[10:]]), -12, 12))
    landing = np.maximum(weights / weights.sum(), 1e-5)
    landing /= landing.sum()
    population = landing / (1 - retention)
    population /= population.sum()
    transition = np.diag(retention) + (1 - retention[:, None]) * landing
    quadratic = np.column_stack(
        (
            centres[:, 0] ** 2 + variance,
            centres[:, 0] * centres[:, 1],
            centres[:, 1] ** 2 + variance,
        )
    )
    values = [*(population @ centres), *(population @ quadratic)]
    early, late = [], []
    power = np.eye(3, dtype=np.longdouble)
    for lag in range(1, 16):
        power = power @ transition
        joint = population[:, None] * power
        values.extend((centres.T @ joint @ centres).ravel())
        if lag <= 4:
            early.extend((quadratic.T @ joint @ centres).ravel())
            late.extend((centres.T @ joint @ quadratic).ravel())
    return np.array([*values, *early, *late], dtype=np.longdouble)


def information_oracle(
    free: FloatArray, pair: NDArray[np.int64], *, correlation_only: bool = False
) -> LongArray:
    """Evaluate the spec with decimal arithmetic before test-only differencing.

    Tiny pair landings can make QND so close to one that even long-double
    rounding pollutes a central difference of information.
    """
    with localcontext() as context:
        context.prec = 50
        x = [Decimal(float(value)) for value in free]
        one = Decimal(1)
        keep = [
            min(max(one / (one + (-v).exp()), Decimal(0.01)), Decimal(0.995))
            for v in x[7:10]
        ]
        weights = [
            min(max(v, Decimal(-12)), Decimal(12)).exp() for v in [Decimal(0), *x[10:]]
        ]
        total = sum(weights)
        land = [max(w / total, Decimal(1e-5)) for w in weights]
        total = sum(land)
        land = [w / total for w in land]
        a, b = pair
        q = sum((x[2 * b + c] - x[2 * a + c]) ** 2 for c in (0, 1)) / (4 * x[6].exp())
        qnd = (
            sum(
                (keep[i] + (1 - keep[i]) * land[i])
                / (keep[i] + (1 - keep[i]) * (land[a] + land[b]))
                for i in pair
            )
            - 1
        )
        kappa = (1 + qnd**2 + q * (1 - qnd**2)) / (2 * qnd)
        z = 1 / (kappa + (kappa**2 - 1).sqrt())
        information = (
            4
            * (qnd - z) ** 2
            * (1 - qnd**2 + qnd * z + 3 * qnd**3 * z - 4 * qnd**2 * z**2)
            / (
                (1 - qnd**2)
                * (1 - qnd * z)
                * (
                    1
                    - qnd**2
                    + 4 * qnd * z
                    - 5 * z**2
                    - 3 * qnd**2 * z**2
                    + 4 * qnd * z**3
                )
            )
        )
    return np.array(
        [str(qnd if correlation_only else information)], dtype=np.longdouble
    )


def finite_difference(
    function: Callable[[FloatArray], FloatArray | LongArray],
    free: FloatArray,
    *,
    kink: bool = False,
) -> FloatArray:
    """Richardson-extrapolated central differences, independent of analytic code."""
    columns = []
    for index in range(len(free)):
        direction = np.zeros_like(free)
        step = 2e-3 if kink else 1e-3
        direction[index] = step
        coarse = (function(free + direction) - function(free - direction)) / (2 * step)
        fine = (function(free + direction / 2) - function(free - direction / 2)) / step
        if kink:
            # At a kink remove linear, quadratic and cubic truncation terms.
            quarter = (
                function(free + direction / 4) - function(free - direction / 4)
            ) / (step / 2)
            eighth = (
                function(free + direction / 8) - function(free - direction / 8)
            ) / (step / 4)
            columns.append((-coarse + 14 * fine - 56 * quarter + 64 * eighth) / 21)
        else:
            columns.append((4 * fine - coarse) / 3)
    return np.asarray(np.column_stack(columns), dtype=np.float64)


def model_cases() -> list[ModelParameters]:
    """Exact-moment reference geometry and reproducible skewed three-state models."""
    models = [
        ModelParameters(
            separation * np.array([[-0.5, 0], [0.5, 0], [0, 1]]),
            1.0,
            np.array([0.75, 0.75, 0.9]),
            np.array([0.49, 0.49, 0.02]),
        )
        for separation in (1.0, 2.0, 3.0)
    ]
    # Additional saved-moment regression cases: an unseen state, skewed centres,
    # retentions near the upper bound, and retentions clipped above that bound.
    references = [
        ([-1.5, 0, 1.5, 0, 0, 0], [0.75, 0.75, 0.9], [0.5, 0.5, 0]),
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
        ),
    ]
    models.extend(
        ModelParameters(
            np.array(centres).reshape(3, 2), 1.0, np.array(retention), np.array(landing)
        )
        for centres, retention, landing in references
    )
    rng = np.random.default_rng(817)
    for _ in range(25):
        landing = np.exp(rng.uniform(-9, 0, 3))
        models.append(
            ModelParameters(
                rng.normal(size=(3, 2)),
                float(np.exp(rng.uniform(-1, 1))),
                rng.uniform(0.02, 0.9949, 3),
                landing / landing.sum(),
            )
        )
    models.append(
        ModelParameters(
            np.array([[-1.5, 0.1], [1.5, -0.2], [0.3, 3.0]]),
            0.7,
            np.array([0.9949, 0.98, 0.994]),
            np.array([0.499988, 0.5, 0.000012]),
        )
    )
    return models


def assert_derivative(actual: FloatArray, expected: FloatArray) -> None:
    """Compare each coordinate relatively; roundoff-only zero columns use 1e-8."""
    error = np.linalg.norm(actual - expected, axis=0)
    scale = np.linalg.norm(expected, axis=0)
    np.testing.assert_array_less(error, 1e-6 * scale + 1e-8)


@pytest.mark.parametrize("model", model_cases())
def test_moment_jacobian_matches_independent_oracle(model: ModelParameters) -> None:
    with np.errstate(divide="ignore"):
        free = _estimator.free_coordinates(model)
    expected = finite_difference(
        moment_oracle,
        free,
        kink=bool(np.any(model.retention == 0.995)),
    )

    actual = _estimator.moment_jacobian(free)

    assert actual.shape == (113, 12)
    assert_derivative(actual, expected)


@pytest.mark.parametrize("model", model_cases())
@pytest.mark.parametrize("pair", [np.array([0, 1]), np.array([2, 0]), np.array([1, 2])])
def test_pair_information_gradient_matches_independent_oracle(
    model: ModelParameters, pair: NDArray[np.int64]
) -> None:
    with np.errstate(divide="ignore"):
        free = _estimator.free_coordinates(model)
    expected = finite_difference(
        lambda value: information_oracle(value, pair),
        free,
        kink=bool(np.any(model.retention == 0.995)),
    )

    actual = _information.pair_information_gradient(free, pair)

    assert actual.shape == (12,)
    assert_derivative(actual[None, :], expected)
    expected_qnd = finite_difference(
        lambda value: information_oracle(value, pair, correlation_only=True),
        free,
        kink=bool(np.any(model.retention == 0.995)),
    )
    assert_derivative(
        _information.pair_correlation_gradient(free, pair)[None, :], expected_qnd
    )


@pytest.mark.parametrize(
    "retention_free, landing_free",
    [
        ([-8.0, 8.0, 0.0], [-15.0, 15.0]),
        ([np.log(0.01 / 0.99), np.log(0.995 / 0.005), 0.0], [0.0, -9.0]),
        ([1.0, 2.0, 3.0], [-12.0, 12.0]),
        ([1.0, 2.0, 3.0], [0.0, -11.0]),
        ([1.0, 2.0, 3.0], [0.0, np.log(2e-5 / (1 - 1e-5))]),
    ],
    ids=[
        "outside-clips",
        "retention-kinks",
        "ratio-kinks",
        "below-floor",
        "floor-kink",
    ],
)
def test_bounded_derivatives_follow_clips_and_mean_one_sided_kink_slopes(
    retention_free: list[float], landing_free: list[float]
) -> None:
    free = np.array(
        [-1.3, 0.2, 1.8, -0.1, 0.4, 3.0, 0.0, *retention_free, *landing_free]
    )
    pair = np.array([0, 1])
    moments = finite_difference(
        moment_oracle,
        free,
        kink=True,
    )
    information = finite_difference(
        lambda value: information_oracle(value, pair),
        free,
        kink=True,
    )

    assert_derivative(_estimator.moment_jacobian(free), moments)
    assert_derivative(
        _information.pair_information_gradient(free, pair)[None, :], information
    )
    qnd = finite_difference(
        lambda value: information_oracle(value, pair, correlation_only=True),
        free,
        kink=True,
    )
    assert_derivative(_information.pair_correlation_gradient(free, pair)[None, :], qnd)


def test_fit_sampling_covariance_excludes_spurious_third_state_from_weights() -> None:
    model = ModelParameters(
        np.array([[-0.7, 0.0], [0.7, 0.0], [0.2, 1.0]]),
        1.0,
        np.array([0.985, 0.95, 0.9]),
        np.array([0.56, 0.439, 0.001]),
    )
    features = moment_means(model)
    blocks = 100_000.0
    start = _estimator.safe_start(features)
    landing = start.landing.copy()
    landing[np.argmin(stationary_population(start))] = 0
    weight_model = start._replace(landing=landing / landing.sum())
    weights = contiguous_covariance(weight_model) / blocks
    values, vectors = np.linalg.eigh((weights + weights.T) / 2)
    whitener = (vectors / np.sqrt(np.maximum(values, values.max() * 1e-14))).T

    free, actual = _estimator.bounded_fit(features, blocks)

    # Independent numerical tangent and the specified fixed-prior sandwich.
    derivative = whitener @ finite_difference(moment_oracle, free)
    _, scale = _estimator.prior_coordinates(start)
    normal = derivative.T @ derivative + np.diag(1 / scale**2)
    response = np.linalg.solve(normal, derivative.T)
    expected = response @ response.T
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-9)
