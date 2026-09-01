from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.experiment.v2.singleshot.t1.t1 import T1Exp, T1Result
from zcu_tools.experiment.v2.singleshot.t1.t1_with_tone import (
    T1WithToneExp,
    T1WithToneResult,
)
from zcu_tools.utils.fitting.multi_decay import model_func


def _synthetic_signals(times: NDArray[np.float64]) -> NDArray[np.float64]:
    rates = (0.1, 0.05, 0.08, 0.04, 0.02, 0.01)
    populations1 = model_func(times, *rates, 0.96, 0.03)
    populations2 = model_func(times, *rates, 0.04, 0.95)
    return np.stack((populations1[:, :2], populations2[:, :2]), axis=1)


def test_t1_analysis_uses_named_transition_fit_result() -> None:
    times = np.linspace(0.0, 20.0, 120)
    result = T1Result(lengths=times, signals=_synthetic_signals(times))

    figure = T1Exp().analyze(result)

    assert isinstance(figure, Figure)
    assert "T_1" in figure.get_suptitle()
    plt.close(figure)


def test_t1_with_tone_analysis_preserves_numeric_and_figure_result() -> None:
    times = np.linspace(0.0, 20.0, 120)
    result = T1WithToneResult(lengths=times, signals=_synthetic_signals(times))

    t1, t1_b, figure = T1WithToneExp().analyze(result)

    assert np.isfinite(t1)
    assert np.isfinite(t1_b)
    assert isinstance(figure, Figure)
    plt.close(figure)
