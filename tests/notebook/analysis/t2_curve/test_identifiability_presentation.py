from __future__ import annotations

import numpy as np
import zcu_tools.notebook.analysis.t2_curve as t2
from matplotlib import pyplot as plt
from zcu_tools.notebook.analysis.fit_tools.identifiability import (
    FitCorrelation,
    ParameterProfile,
    WeakParameter,
)


def test_profile_plot_scales_flux_noise_and_marks_threshold_best_and_bounds() -> None:
    profile = ParameterProfile(
        "A_phi",
        np.array([1e-6, 2e-6, 3e-6]),
        np.array([2.0, 0.0, 2.0]),
        2e-6,
        (1e-6, 3e-6),
        (1.5e-6, 2.5e-6),
        False,
        False,
        None,
        None,
    )
    fig, ax = t2.plot_t2_profile(profile)
    try:
        assert "µΦ0/√Hz" in ax.get_xlabel()
        assert "Δχ²" in ax.get_ylabel()
        np.testing.assert_allclose(
            np.asarray(ax.lines[0].get_xdata(), dtype=np.float64), [1, 2, 3]
        )
        assert {"Δχ² = 1", "Best fit", "Lower bound", "Upper bound"} <= {
            line.get_label() for line in ax.lines
        }
    finally:
        plt.close(fig)


def test_weak_parameter_table_preserves_each_reason() -> None:
    table = t2.make_t2_weak_parameter_table(
        (WeakParameter("A_phi", ("correlation with n_th", "lower profile unbounded")),)
    )
    assert table["parameter"].tolist() == ["A_phi"]
    assert "correlation with n_th" in table.iloc[0]["reasons"]
    assert "lower profile unbounded" in table.iloc[0]["reasons"]


def test_correlation_plot_displays_log_space_and_physical_labels() -> None:
    info = FitCorrelation(("A_phi", "n_th"), np.eye(2), np.eye(2), 2, 1.0, 1.0)
    fig, ax = t2.plot_t2_correlation(info)
    try:
        assert "log" in ax.get_title().lower()
        assert len(ax.texts) == 4
        assert "µΦ0/√Hz" in ax.get_xticklabels()[0].get_text()
    finally:
        plt.close(fig)
