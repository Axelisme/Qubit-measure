from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.notebook.analysis.t1_curve.base import (
    find_proper_Temp,
    plot_eff_t1_with_sample,
)


def test_find_proper_temp_passes_scalar_candidate_to_callback() -> None:
    seen: list[float] = []

    def _calc_Q(temp: float) -> np.ndarray:
        assert isinstance(temp, float)
        seen.append(temp)
        return np.array([temp, 2.0 * temp], dtype=np.float64)

    result = find_proper_Temp(60e-3, _calc_Q, Temp_bounds=(None, 120e-3))

    assert seen
    assert isinstance(result, float)


def test_plot_eff_t1_with_sample_draws_component_limits() -> None:
    fig, ax = plot_eff_t1_with_sample(
        np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        np.array([8.0, 10.0, 9.0], dtype=np.float64),
        np.array([0.5, 0.6, 0.5], dtype=np.float64),
        np.array([7.0, 8.0, 7.5], dtype=np.float64),
        flux_half=0.0,
        flux_period=2.0,
        t_fluxs=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        component_t1s={
            "capacitive": np.array([20.0, 21.0, 22.0], dtype=np.float64),
            "inductive": np.array([30.0, 31.0, 32.0], dtype=np.float64),
        },
        component_bands={
            "capacitive bounds": (
                np.array([18.0, 19.0, 20.0], dtype=np.float64),
                np.array([22.0, 23.0, 24.0], dtype=np.float64),
            ),
        },
        parameter_text="Q_cap = 1.000e+05",
    )

    try:
        _, labels = ax.get_legend_handles_labels()
        lines = {line.get_label(): line for line in ax.lines}
        assert "capacitive" in labels
        assert "inductive" in labels
        assert r"$t_1^{eff}$" in labels
        assert "capacitive bounds" not in labels
        assert labels.index(r"$t_1^{eff}$") < labels.index("capacitive")
        assert lines["capacitive"].get_linestyle() == ":"
        assert lines[r"$t_1^{eff}$"].get_linestyle() == "-"
        assert "Q_cap = 1.000e+05" in {text.get_text() for text in ax.texts}
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_plot_eff_t1_with_sample_keeps_sample_ylim_with_large_component() -> None:
    fig, ax = plot_eff_t1_with_sample(
        np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        np.array([8.0, 10.0, 9.0], dtype=np.float64),
        np.array([0.5, 0.6, 0.5], dtype=np.float64),
        np.array([7.0, 8.0, 7.5], dtype=np.float64),
        flux_half=0.0,
        flux_period=2.0,
        t_fluxs=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        component_t1s={
            "Purcell": np.array([1000.0, 1100.0, 1200.0], dtype=np.float64),
        },
    )

    try:
        _, labels = ax.get_legend_handles_labels()
        lines = {line.get_label(): line for line in ax.lines}
        assert "Purcell" in labels
        assert lines["Purcell"].get_linestyle() == "-."
        assert ax.get_ylim()[1] < 1200.0
    finally:
        plt.close(fig)


def test_plot_eff_t1_with_sample_rejects_component_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="component_t1s\\['bad'\\]"):
        fig, _ = plot_eff_t1_with_sample(
            np.array([-1.0, 0.0, 1.0], dtype=np.float64),
            np.array([8.0, 10.0, 9.0], dtype=np.float64),
            np.array([0.5, 0.6, 0.5], dtype=np.float64),
            np.array([7.0, 8.0, 7.5], dtype=np.float64),
            flux_half=0.0,
            flux_period=2.0,
            t_fluxs=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            component_t1s={"bad": np.array([20.0, 21.0], dtype=np.float64)},
        )
        plt.close(fig)


def test_plot_eff_t1_with_sample_rejects_band_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="component_bands\\['bad'\\]"):
        fig, _ = plot_eff_t1_with_sample(
            np.array([-1.0, 0.0, 1.0], dtype=np.float64),
            np.array([8.0, 10.0, 9.0], dtype=np.float64),
            np.array([0.5, 0.6, 0.5], dtype=np.float64),
            np.array([7.0, 8.0, 7.5], dtype=np.float64),
            flux_half=0.0,
            flux_period=2.0,
            t_fluxs=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            component_bands={
                "bad": (
                    np.array([20.0, 21.0], dtype=np.float64),
                    np.array([22.0, 23.0], dtype=np.float64),
                )
            },
        )
        plt.close(fig)
