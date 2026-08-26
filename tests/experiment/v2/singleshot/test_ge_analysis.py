from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.ge import GE_Exp, GE_Result


def _result() -> GE_Result:
    return GE_Result(
        signals=np.array(
            [
                [-1.0 + 0.0j, -0.9 + 0.0j, 0.9 + 0.0j],
                [1.0 + 0.0j, 0.9 + 0.0j, -0.9 + 0.0j],
            ],
            dtype=np.complex128,
        ),
        shot_indices=np.arange(3, dtype=np.int64),
        prepared_states=np.array([0, 1], dtype=np.int64),
    )


def test_confusion_calculation_is_figure_free_and_returns_render_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_figure(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("numeric confusion calculation created a figure")

    monkeypatch.setattr(
        "zcu_tools.experiment.v2.singleshot.ge.plt.subplots", reject_figure
    )

    result = GE_Exp().calc_confusion_matrix(
        np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
        -1.0 + 0.0j,
        1.0 + 0.0j,
        sigma=0.2,
        radius=0.3,
        result=_result(),
        consider_other=False,
    )

    assert result.radius == pytest.approx(0.3)
    assert result.init_matrix.shape == (3, 3)
    assert result.matrix.shape == (3, 3)
    assert result.g_classification == pytest.approx((2 / 3, 1 / 3, 0.0))
    assert result.e_classification == pytest.approx((1 / 3, 2 / 3, 0.0))
    assert result.condition_number == pytest.approx(np.linalg.cond(result.matrix))


def test_confusion_renderer_preserves_diagnostic_content() -> None:
    exp = GE_Exp()
    run_result = _result()
    confusion = exp.calc_confusion_matrix(
        np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
        -1.0 + 0.0j,
        1.0 + 0.0j,
        sigma=0.2,
        radius=0.3,
        result=run_result,
        consider_other=False,
    )

    figure = exp.plot_confusion_matrix(
        confusion,
        -1.0 + 0.0j,
        1.0 + 0.0j,
        result=run_result,
    )
    try:
        titled_axes = {
            axis.get_title(): axis for axis in figure.axes if axis.get_title()
        }
        assert len(figure.axes) == 6  # four panels plus two colorbars
        assert "Initial Populations" in titled_axes
        confusion_title = next(
            title
            for title in titled_axes
            if title.startswith("Confusion Matrix (cond:")
        )
        assert confusion_title == (
            f"Confusion Matrix (cond: {confusion.condition_number:.1f})"
        )
        assert len(titled_axes["Initial Populations"].texts) == 9
        assert len(titled_axes[confusion_title].texts) == 9
        assert titled_axes[confusion_title].get_xlabel() == "Measured State"
        assert titled_axes[confusion_title].get_ylabel() == "Actual State"
        prepared_titles = [title for title in titled_axes if title.startswith("$|0")]
        assert len(prepared_titles) == 2
        assert all("66.7%" in title and "33.3%" in title for title in prepared_titles)
    finally:
        plt.close(figure)
