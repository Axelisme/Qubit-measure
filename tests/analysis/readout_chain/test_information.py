"""All-lag information at the numerical information seam; oracles stay here."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.analysis.readout_chain._information import (
    all_lag_information,
    all_lag_information_gradient,
    pair_information,
)
from zcu_tools.analysis.readout_chain._model import ModelParameters


def truncated_information(q: float, correlation: float) -> float:
    """Independent GLS solve of exact symmetric-chain lag-product covariance."""
    count = int(np.ceil(20 / -np.log(correlation)))
    lag = np.arange(1, count + 1, dtype=float)
    left, right = np.meshgrid(lag, lag, indexing="ij")
    difference, total = np.abs(left - right), left + right
    tail = (1 + correlation**2) / (1 - correlation**2)
    covariance = (
        np.eye(count)
        + 2 * q * (correlation**difference + correlation**total)
        + q**2
        * (
            (difference + tail) * correlation**difference
            - (total + tail) * correlation**total
        )
    )
    response = -2 * lag * q * correlation**lag
    return float(response @ np.linalg.solve(covariance, response))


def symmetric_model(q: float, correlation: float) -> ModelParameters:
    """Embed a symmetric pair in a three-state model with no third landing."""
    return ModelParameters(
        np.array([[-np.sqrt(q), 0.0], [np.sqrt(q), 0.0], [0.0, 3.0]]),
        1.0,
        np.array([correlation, correlation, 0.9]),
        np.array([0.5, 0.5, 0.0]),
    )


@pytest.mark.slow
@pytest.mark.parametrize("q", [0.0225, 0.25, 2.25, 9.0])
@pytest.mark.parametrize("correlation", [0.3, 0.7, 0.9, 0.98, 0.995])
def test_all_lag_closed_form_matches_truncated_gls(
    q: float, correlation: float
) -> None:
    actual = pair_information(symmetric_model(q, correlation), np.array([0, 1]))

    assert_allclose(actual, truncated_information(q, correlation), rtol=1e-9)


@pytest.mark.parametrize("q", [0.0225, 0.25, 2.25, 9.0])
@pytest.mark.parametrize("correlation", [0.3, 0.7, 0.9, 0.98, 0.995])
def test_q_and_correlation_gradient_matches_finite_difference(
    q: float, correlation: float
) -> None:
    expected = []
    point = np.array([q, correlation])
    for index in range(2):
        direction = np.zeros(2)
        step = 1e-5 * min(q, 1 - correlation)
        direction[index] = step
        coarse = (
            all_lag_information(*(point + direction))
            - all_lag_information(*(point - direction))
        ) / (2 * step)
        fine = (
            all_lag_information(*(point + direction / 2))
            - all_lag_information(*(point - direction / 2))
        ) / step
        expected.append((4 * fine - coarse) / 3)

    assert_allclose(all_lag_information_gradient(q, correlation), expected, rtol=1e-6)


@pytest.mark.parametrize("correlation", [0.3, 0.9, 0.995])
def test_noiseless_limit_is_complete_state_information(correlation: float) -> None:
    actual = pair_information(symmetric_model(1e10, correlation), np.array([0, 1]))

    assert_allclose(actual, 4 * correlation**2 / (1 - correlation**2), rtol=1e-6)
