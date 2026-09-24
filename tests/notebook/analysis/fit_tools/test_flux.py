from __future__ import annotations

import warnings

import numpy as np
import pytest
import zcu_tools.notebook.analysis.fit_tools.flux as flux_mod
from zcu_tools.notebook.analysis.fit_tools import (
    align_flux_to_window,
    correct_flux_from_f01,
    predict_f01_mhz,
)


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


def test_correct_flux_from_f01_tie_prefers_periodic_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_solve(
        freq_mhz: float,
        params: tuple[float, float, float],
        *,
        guess_flux: float,
    ) -> float:
        assert params == (3.0, 1.0, 0.5)
        # Direct candidate such that periodic and mirror branches tie at 0.1
        # distance from the raw flux.
        return guess_flux - 0.1

    monkeypatch.setattr(flux_mod, "_solve_f01_candidate_flux", fake_solve)

    result = correct_flux_from_f01(
        np.array([0.5], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        (3.0, 1.0, 0.5),
        max_abs_flux_correction=0.11,
    )

    # periodic 0.4 and mirror 0.6 are both 0.1 away; the deterministic tie rule
    # keeps the periodic branch.
    np.testing.assert_allclose(result.corrected_fluxs, [0.4])
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


def test_solve_f01_candidate_flux_rejects_unreachable_frequencies() -> None:
    # Probe frequencies far outside the model f01 range must not silently fall
    # back to the guessed flux (that would read as a zero, apparently
    # successful, correction).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        candidates = [
            flux_mod._solve_f01_candidate_flux(
                1e3 * freq_ghz, (3.0, 1.0, 0.5), guess_flux=0.5
            )
            for freq_ghz in (-1.0, 0.0, 100.0)
        ]
    assert all(not np.isfinite(candidate) for candidate in candidates)


def test_correct_flux_from_f01_rejects_non_converged_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_solve(
        freq_mhz: float,
        params: tuple[float, float, float],
        *,
        guess_flux: float,
    ) -> float:
        assert params == (3.0, 1.0, 0.5)
        if freq_mhz < 0.0:
            return np.nan  # non-converged solve produces no candidate
        return guess_flux + 0.01

    monkeypatch.setattr(flux_mod, "_solve_f01_candidate_flux", fake_solve)

    result = correct_flux_from_f01(
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([-1.0, 1.01], dtype=np.float64),
        (3.0, 1.0, 0.5),
        max_abs_flux_correction=0.03,
    )

    np.testing.assert_allclose(result.corrected_fluxs, [0.5, 0.51])
    np.testing.assert_array_equal(result.accepted, [False, True])
    np.testing.assert_allclose(result.applied_flux_corrections, [0.0, 0.01])
    assert result.skipped_count == 1


def test_correct_flux_from_f01_end_to_end_rejects_unreachable_frequencies() -> None:
    # Regression: -1/0/100 GHz probes used to be accepted with a zero
    # correction because the failed solve returned the raw guess.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = correct_flux_from_f01(
            np.array([0.5, 0.5, 0.5], dtype=np.float64),
            np.array([-1.0, 0.0, 100.0], dtype=np.float64),
            (3.4, 0.9, 0.6),
            max_abs_flux_correction=0.03,
        )

    np.testing.assert_array_equal(result.accepted, [False, False, False])
    np.testing.assert_allclose(result.corrected_fluxs, [0.5, 0.5, 0.5])
    assert result.skipped_count == 3


def test_correct_flux_from_f01_converges_on_reachable_frequency() -> None:
    params = (3.4, 0.9, 0.6)
    raw_fluxs = np.array([0.5], dtype=np.float64)
    f01_mhz = predict_f01_mhz(params, raw_fluxs)

    result = correct_flux_from_f01(
        raw_fluxs, 1e-3 * f01_mhz, params, max_abs_flux_correction=0.03
    )

    np.testing.assert_array_equal(result.accepted, [True])
    np.testing.assert_allclose(result.corrected_fluxs, [0.5])


def test_align_flux_to_window_aligns_integer_equivalent_branches() -> None:
    fluxs = np.array([0.5, -0.5, 1.5], dtype=np.float64)

    aligned, shifts, in_window = align_flux_to_window(fluxs, (0.49, 0.53))

    np.testing.assert_allclose(aligned, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(shifts, [0.0, 1.0, -1.0])
    np.testing.assert_array_equal(in_window, [True, True, True])


def test_align_flux_to_window_tie_picks_smaller_resulting_flux() -> None:
    # Window midpoint 0.0; flux 0.5 has 0.5 and -0.5 at equal distance, and the
    # deterministic tie rule picks the smaller resulting flux (-0.5).
    aligned, shifts, in_window = align_flux_to_window(
        np.array([0.5], dtype=np.float64), (-0.2, 0.2)
    )

    np.testing.assert_allclose(aligned, [-0.5])
    np.testing.assert_allclose(shifts, [-1.0])
    np.testing.assert_array_equal(in_window, [False])


def test_align_flux_to_window_out_of_window_rows_are_excluded() -> None:
    aligned, shifts, in_window = align_flux_to_window(
        np.array([0.5, -0.5], dtype=np.float64), (0.49, 0.53)
    )

    np.testing.assert_allclose(aligned, [0.5, 0.5])
    np.testing.assert_allclose(shifts, [0.0, 1.0])
    np.testing.assert_array_equal(in_window, [True, True])


def test_align_flux_to_window_rejects_invalid_windows() -> None:
    with pytest.raises(ValueError, match="finite"):
        align_flux_to_window(np.array([0.5]), (np.nan, 0.6))
    with pytest.raises(ValueError, match="strictly increasing"):
        align_flux_to_window(np.array([0.5]), (0.6, 0.6))
    with pytest.raises(ValueError, match="strictly increasing"):
        align_flux_to_window(np.array([0.5]), (0.6, 0.5))
    with pytest.raises(ValueError, match="one flux period"):
        align_flux_to_window(np.array([0.5]), (0.0, 1.1))


def test_align_flux_to_window_is_deterministic() -> None:
    fluxs = np.array([-0.5, 0.5, 1.5], dtype=np.float64)
    first = align_flux_to_window(fluxs, (0.48, 0.54))
    second = align_flux_to_window(fluxs.copy(), (0.48, 0.54))
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
        )
