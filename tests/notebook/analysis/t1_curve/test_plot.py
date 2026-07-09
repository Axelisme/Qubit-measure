from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.notebook.analysis.t1_curve.base import plot_eff_t1_with_sample


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
        parameter_text="Q_cap = 1.000e+05",
    )

    try:
        _, labels = ax.get_legend_handles_labels()
        assert "capacitive" in labels
        assert "inductive" in labels
        assert r"$t_1^{eff}$" in labels
        assert "Q_cap = 1.000e+05" in {text.get_text() for text in ax.texts}
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
