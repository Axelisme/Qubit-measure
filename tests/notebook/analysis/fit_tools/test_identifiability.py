from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.notebook.analysis.fit_tools import (
    FitCorrelation,
    calculate_fit_correlation,
    find_weak_parameters,
)


@pytest.mark.parametrize(
    ("pairwise", "log_std", "reason_counts"),
    [(0.96, 0.2, (1, 1)), (0.96, 2.6, (2, 2)), (0.95, 1.0, ())],
)
def test_weak_parameter_rule_reports_all_and_only_strict_threshold_exceedances(
    pairwise: float, log_std: float, reason_counts: tuple[int, ...]
) -> None:
    info = FitCorrelation(
        ("A_phi", "n_th"),
        np.eye(2) * log_std**2,
        np.array([[1.0, -pairwise], [-pairwise, 1.0]]),
        2,
        4.0,
        1.0,
    )
    weak = find_weak_parameters(info)
    assert tuple(len(entry.reasons) for entry in weak) == reason_counts
    if reason_counts:
        assert [entry.parameter for entry in weak] == ["A_phi", "n_th"]
        assert "n_th" in weak[0].reasons[0]
        assert "0.96" in weak[0].reasons[0]


@pytest.mark.parametrize("cost", [0.0, 1.0])
def test_zero_jacobian_column_is_unconstrained_even_with_zero_cost(cost: float) -> None:
    info = calculate_fit_correlation(
        np.array([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        cost,
        3,
        ("measured", "unconstrained"),
    )
    assert info is not None
    assert info.rank == 1
    assert info.condition_number == np.inf
    assert info.covariance[0, 0] == pytest.approx(cost / 2)
    assert info.covariance[1, 1] == np.inf
    assert info.correlation[0, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(info.correlation[1, :]))
    assert np.all(np.isnan(info.correlation[:, 1]))
    weak = find_weak_parameters(info)
    assert [entry.parameter for entry in weak] == ["unconstrained"]
    assert "sigma(ln unconstrained) = inf" in weak[0].reasons[0]
    assert "not constrained by the data" in weak[0].reasons[0]


def test_tiny_independent_column_has_large_variance_not_false_correlation() -> None:
    # The columns have norms 2 and 1e-12 but are orthogonal, not degenerate.
    info = calculate_fit_correlation(
        np.array([[2.0, 0.0], [0.0, 1e-12], [0.0, 0.0]]),
        0.5,
        3,
        ("measured", "tiny"),
    )
    assert info is not None
    assert info.rank == 2
    assert info.condition_number == pytest.approx(1.0)
    np.testing.assert_allclose(info.covariance, np.diag([0.25, 1e24]))
    np.testing.assert_allclose(info.correlation, np.eye(2))
    assert [entry.parameter for entry in find_weak_parameters(info)] == ["tiny"]


@pytest.mark.parametrize("rows", [1, 4])
def test_dependent_parameters_are_unconstrained_without_poisoning_independent_one(
    rows: int,
) -> None:
    # a + 2*b is measured, but neither a nor b alone is determined.
    jacobian = np.array(
        [[1.0, 2.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )[:rows]
    info = calculate_fit_correlation(jacobian, 0.5, 4, ("a", "b", "c"))
    assert info is not None
    assert info.rank == min(rows, 2)
    assert info.condition_number == np.inf
    assert np.all(np.isinf(np.diag(info.covariance)[:2]))
    assert np.all(np.isnan(info.correlation[:2, :]))
    assert np.all(np.isnan(info.correlation[:, :2]))
    if rows > 1:
        assert info.covariance[2, 2] == pytest.approx(1 / 16)
        assert info.correlation[2, 2] == pytest.approx(1.0)
        assert [entry.parameter for entry in find_weak_parameters(info)] == ["a", "b"]
    else:
        assert info.covariance[2, 2] == np.inf
        assert [entry.parameter for entry in find_weak_parameters(info)] == [
            "a",
            "b",
            "c",
        ]


@pytest.mark.parametrize(
    ("jacobian", "cost"),
    [
        (np.array([[np.nan], [1.0]]), 1.0),
        (np.array([[np.inf], [1.0]]), 1.0),
        (np.ones((2, 1)), np.nan),
        (np.ones((2, 1)), np.inf),
    ],
)
def test_nonfinite_optimizer_geometry_provides_no_matrix(
    jacobian: NDArray[np.float64], cost: float
) -> None:
    assert calculate_fit_correlation(jacobian, cost, 2, ("a",)) is None


def test_missing_matrix_provides_no_weak_parameter_evidence() -> None:
    assert find_weak_parameters(None) == ()


def test_large_log_uncertainty_is_weak_without_pairwise_correlation() -> None:
    info = FitCorrelation(
        ("A_phi", "n_th"), np.diag([0.01, 6.76]), np.eye(2), 2, 1.0, 1.0
    )
    weak = find_weak_parameters(info)
    assert [entry.parameter for entry in weak] == ["n_th"]
    assert "2.6" in weak[0].reasons[0]
    assert "1" in weak[0].reasons[0]
