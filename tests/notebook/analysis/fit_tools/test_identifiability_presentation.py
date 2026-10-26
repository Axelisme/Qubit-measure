from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from zcu_tools.notebook.analysis.fit_tools import (
    FitCorrelation,
    WeakParameter,
    make_weak_parameter_table,
    plot_fit_correlation,
)


def test_correlation_plot_displays_parameter_order_values_and_log_space() -> None:
    info = FitCorrelation(
        ("A_phi", "n_th"),
        np.eye(2),
        np.array([[1.0, -0.97], [-0.97, 1.0]]),
        2,
        8.0,
        1.0,
    )
    fig, ax = plot_fit_correlation(info)
    try:
        assert "log" in ax.get_title().lower()
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["A_phi", "n_th"]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["A_phi", "n_th"]
        assert [text.get_text() for text in ax.texts] == [
            "1.00",
            "-0.97",
            "-0.97",
            "1.00",
        ]
    finally:
        plt.close(fig)


def test_weak_parameter_table_preserves_all_reasons() -> None:
    table = make_weak_parameter_table(
        (
            WeakParameter(
                "n_th", ("correlation with A_phi = 0.97", "sigma(ln n_th) = 2.6")
            ),
        )
    )
    assert table["parameter"].tolist() == ["n_th"]
    assert "correlation with A_phi = 0.97" in table.iloc[0]["reasons"]
    assert "sigma(ln n_th) = 2.6" in table.iloc[0]["reasons"]


def test_no_weak_parameters_produces_empty_table_with_named_columns() -> None:
    table = make_weak_parameter_table(())
    assert table.empty
    assert table.columns.tolist() == ["parameter", "reasons"]
