from __future__ import annotations

import numpy as np
import pytest
import zcu_tools.notebook.analysis.fit_tools.flux as flux_mod
from zcu_tools.notebook.analysis.t1_curve import correct_flux_from_f01


def _monkeypatch_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[float, float]]:
    """Install a deterministic per-row solver and record (freq_mhz, guess) calls."""
    calls: list[tuple[float, float]] = []

    def fake_solve(
        freq_mhz: float,
        params: tuple[float, float, float],
        *,
        guess_flux: float,
    ) -> float:
        calls.append((freq_mhz, guess_flux))
        assert params == (3.0, 1.0, 0.5)
        return guess_flux + (freq_mhz - 1000.0) / 1000.0

    monkeypatch.setattr(flux_mod, "_solve_f01_candidate_flux", fake_solve)
    return calls


def test_correct_flux_from_f01_takes_ghz_and_converts_to_mhz_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _monkeypatch_solver(monkeypatch)

    result = correct_flux_from_f01(
        np.array([0.5, 1.5, 2.5], dtype=np.float64),
        np.array([1.01, 1.10, 1.02], dtype=np.float64),
        (3.0, 1.0, 0.5),
        max_abs_flux_correction=0.03,
    )

    # The MHz-based predictor must receive the explicit GHz -> MHz conversion.
    np.testing.assert_allclose([call[0] for call in calls], [1010.0, 1100.0, 1020.0])
    np.testing.assert_allclose(result.raw_fluxs, [0.5, 1.5, 2.5])
    np.testing.assert_allclose(result.corrected_fluxs, [0.51, 1.5, 2.52])
    np.testing.assert_array_equal(result.accepted, [True, False, True])
    np.testing.assert_allclose(result.applied_flux_corrections, [0.01, 0.0, 0.02])
    assert result.skipped_count == 1


def test_correct_flux_from_f01_uses_nearest_mirror_equivalent_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _monkeypatch_solver(monkeypatch)

    # Direct candidate 0.51 would be 0.02 away, but the mirror-equivalent
    # 0.49 is exactly the raw flux, so it must be chosen.
    result = correct_flux_from_f01(
        np.array([0.49], dtype=np.float64),
        np.array([1.02], dtype=np.float64),
        (3.0, 1.0, 0.5),
        max_abs_flux_correction=0.03,
    )

    np.testing.assert_allclose(result.corrected_fluxs, [0.49])
    np.testing.assert_array_equal(result.accepted, [True])


def test_correct_flux_from_f01_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        correct_flux_from_f01(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            (3.0, 1.0, 0.5),
        )

    with pytest.raises(ValueError, match="one-dimensional"):
        correct_flux_from_f01(
            np.array([[0.0]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            (3.0, 1.0, 0.5),
        )


def test_correct_flux_from_f01_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        correct_flux_from_f01(
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            (3.0, 1.0, 0.5),
            max_abs_flux_correction=-1.0,
        )


def test_correct_flux_from_f01_missing_frequency_keeps_raw_flux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _monkeypatch_solver(monkeypatch)

    result = correct_flux_from_f01(
        np.array([0.5], dtype=np.float64),
        np.array([np.nan], dtype=np.float64),
        (3.0, 1.0, 0.5),
    )

    assert not np.isfinite(calls[0][0])
    np.testing.assert_allclose(result.corrected_fluxs, [0.5])
    np.testing.assert_array_equal(result.accepted, [False])
