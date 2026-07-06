"""Focused tests for readout-length optimization analysis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import zcu_tools.experiment.v2.twotone.ro_optimize.length as length_mod
from zcu_tools.experiment.v2.twotone.ro_optimize.length import LengthExp, LengthResult


def test_length_analyze_t0_zero_uses_smoothed_snr_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths = np.array([1.0, 4.0, 9.0], dtype=np.float64)
    signals = np.array([5.0, 9.0, 10.0], dtype=np.float64)
    smoothed = np.array([5.0, 9.0, 10.0], dtype=np.float64)

    def _smooth_signal1d(*args: object, **kwargs: object) -> np.ndarray:
        return smoothed.copy()

    monkeypatch.setattr(length_mod, "smooth_signal1d", _smooth_signal1d)

    exp = LengthExp()
    no_penalty_length, fig = exp.analyze(
        LengthResult(lengths=lengths, signals=signals), t0=None
    )
    plt.close(fig)
    zero_penalty_length, fig = exp.analyze(
        LengthResult(lengths=lengths, signals=signals), t0=0.0
    )
    plt.close(fig)
    positive_penalty_length, fig = exp.analyze(
        LengthResult(lengths=lengths, signals=signals), t0=1.0
    )
    plt.close(fig)

    assert no_penalty_length == pytest.approx(9.0)
    assert zero_penalty_length == pytest.approx(9.0)
    assert positive_penalty_length == pytest.approx(4.0)


def test_length_analyze_rejects_negative_t0() -> None:
    result = LengthResult(
        lengths=np.array([1.0, 2.0], dtype=np.float64),
        signals=np.array([1.0, 2.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="t0 length penalty must be non-negative"):
        LengthExp().analyze(result, t0=-0.1)
