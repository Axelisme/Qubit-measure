from __future__ import annotations

import numpy as np
import zcu_tools.notebook.analysis.fit_tools.identifiability as diagnostics


def test_weak_parameters_report_correlation_and_each_open_profile_side() -> None:
    info = diagnostics.FitCorrelation(
        ("A_phi", "n_th"),
        np.eye(2),
        np.array([[1, -0.96], [-0.96, 1]]),
        2,
        7.0,
        1.0,
    )
    profile = diagnostics.ParameterProfile(
        "n_th",
        np.array([0.001, 0.01, 0.1]),
        np.array([0.5, 0.0, 0.2]),
        0.01,
        (0.001, 1.0),
        (None, None),
        True,
        True,
        "bound",
        "scan range",
    )
    weak = diagnostics.find_weak_parameters(info, (profile,))
    assert [entry.parameter for entry in weak] == ["A_phi", "n_th"]
    assert "correlation" in weak[0].reasons[0]
    assert "n_th" in weak[0].reasons[0]
    assert any("lower" in reason and "bound" in reason for reason in weak[1].reasons)
    assert any(
        "upper" in reason and "scan range" in reason for reason in weak[1].reasons
    )
