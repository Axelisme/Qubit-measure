from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.notebook.analysis.fit_tools import FitCorrelation, find_weak_parameters


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
