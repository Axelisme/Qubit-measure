from __future__ import annotations

import numpy as np
import pytest
import zcu_tools.notebook.analysis.t1_curve.utils as utils_mod
from zcu_tools.notebook.analysis.t1_curve import correct_flux_from_f01


class _FakePredictor:
    calls: list[tuple[float, float, tuple[int, int]]] = []

    def __init__(
        self,
        params: tuple[float, float, float],
        flux_half: float,
        flux_period: float,
        flux_bias: float,
    ) -> None:
        assert params == (3.0, 1.0, 0.5)
        assert flux_half == 0.0
        assert flux_period == 1.0
        assert flux_bias == 0.0

    def calculate_bias(
        self,
        cur_value: float,
        cur_freq: float,
        transition: tuple[int, int] = (0, 1),
    ) -> float:
        self.calls.append((cur_value, cur_freq, transition))
        return (cur_freq - 1000.0) / 1000.0


def test_correct_flux_from_f01_applies_small_biases_and_rejects_large_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakePredictor.calls = []
    monkeypatch.setattr(utils_mod, "FluxoniumPredictor", _FakePredictor)

    result = correct_flux_from_f01(
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
        np.array([1.01, 1.10, 1.02], dtype=np.float64),
        (3.0, 1.0, 0.5),
        flux_half=0.0,
        flux_period=1.0,
        max_abs_flux_correction=0.03,
    )

    np.testing.assert_allclose(result.raw_fluxs, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(result.candidate_biases, [0.01, 0.10, 0.02])
    np.testing.assert_allclose(result.candidate_flux_corrections, [0.01, 0.10, 0.02])
    np.testing.assert_array_equal(result.accepted, [True, False, True])
    np.testing.assert_allclose(result.corrected_fluxs, [0.51, 1.5, 2.52])
    np.testing.assert_allclose(result.corrected_dev_values, [0.01, 1.0, 2.02])
    np.testing.assert_allclose(result.applied_flux_corrections, [0.01, 0.0, 0.02])
    assert result.skipped_count == 1
    assert _FakePredictor.calls == [
        (0.0, 1010.0, (0, 1)),
        (1.0, 1100.0, (0, 1)),
        (2.0, 1020.0, (0, 1)),
    ]


def test_correct_flux_from_f01_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        correct_flux_from_f01(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            (3.0, 1.0, 0.5),
            0.0,
            1.0,
        )

    with pytest.raises(ValueError, match="one-dimensional"):
        correct_flux_from_f01(
            np.array([[0.0]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            (3.0, 1.0, 0.5),
            0.0,
            1.0,
        )


def test_correct_flux_from_f01_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        correct_flux_from_f01(
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            (3.0, 1.0, 0.5),
            0.0,
            1.0,
            max_abs_flux_correction=-1.0,
        )
