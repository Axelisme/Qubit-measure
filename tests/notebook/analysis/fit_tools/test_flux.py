from __future__ import annotations

import warnings

import numpy as np
import pytest
from zcu_tools.notebook.analysis.fit_tools import (
    align_flux_to_window,
    correct_flux_from_f01,
    predict_f01_mhz,
)


def test_correct_flux_from_f01_takes_ghz_and_converts_to_mhz_predictor() -> None:
    params = (3.4, 0.9, 0.6)
    raw_fluxs = np.array([0.3, 1.3, 2.3], dtype=np.float64)
    target_fluxs = np.array([0.31, 1.4, 2.32], dtype=np.float64)
    freqs_ghz = 1e-3 * predict_f01_mhz(params, target_fluxs)

    result = correct_flux_from_f01(
        raw_fluxs, freqs_ghz, params, max_abs_flux_correction=0.03
    )

    # Real model frequencies exercise units, periodic branches and threshold rejection.
    np.testing.assert_allclose(result.raw_fluxs, raw_fluxs)
    np.testing.assert_allclose(result.corrected_fluxs, [0.31, 1.3, 2.32], atol=1e-5)
    np.testing.assert_array_equal(result.accepted, [True, False, True])
    np.testing.assert_allclose(
        result.applied_flux_corrections, [0.01, 0.0, 0.02], atol=1e-5
    )
    assert result.skipped_count == 1


def test_correct_flux_from_f01_uses_nearest_mirror_equivalent_branch() -> None:
    params = (3.4, 0.9, 0.6)
    raw_fluxs = np.array([0.49, 0.51], dtype=np.float64)
    target_fluxs = np.array([0.495, 0.505], dtype=np.float64)
    # Both rows have the same physical frequency but different nearest branches.
    freqs_ghz = np.repeat(
        1e-3 * predict_f01_mhz(params, np.array([0.495], dtype=np.float64)), 2
    )

    result = correct_flux_from_f01(raw_fluxs, freqs_ghz, params)

    np.testing.assert_allclose(result.corrected_fluxs, target_fluxs, atol=1e-5)
    np.testing.assert_array_equal(result.accepted, [True, True])
    np.testing.assert_allclose(
        result.applied_flux_corrections, [0.005, -0.005], atol=1e-5
    )


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


@pytest.mark.parametrize("missing_column", [True, False])
def test_correct_flux_from_f01_missing_frequency_uses_model(
    missing_column: bool,
) -> None:
    result = correct_flux_from_f01(
        np.array([0.51, -0.49], dtype=np.float64),
        None if missing_column else np.array([np.nan, np.nan], dtype=np.float64),
        (3.0, 1.0, 0.5),
    )

    np.testing.assert_allclose(result.raw_fluxs, [0.51, -0.49])
    np.testing.assert_allclose(result.corrected_fluxs, [0.51, -0.49], atol=1e-5)
    np.testing.assert_array_equal(result.accepted, [True, True])
    np.testing.assert_allclose(result.applied_flux_corrections, [0.0, 0.0], atol=1e-5)
    assert result.skipped_count == 0


def test_correct_flux_from_f01_rejects_unreachable_frequencies() -> None:
    # Unreachable frequencies must not read as successful zero corrections.
    raw_fluxs = np.full(3, 0.5, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = correct_flux_from_f01(
            raw_fluxs,
            np.array([-1.0, 0.0, 100.0]),
            (3.0, 1.0, 0.5),
        )
    np.testing.assert_array_equal(result.accepted, [False, False, False])
    np.testing.assert_array_equal(result.corrected_fluxs, raw_fluxs)


def test_correct_flux_from_f01_failed_row_does_not_affect_reachable_row() -> None:
    params = (3.4, 0.9, 0.6)
    reachable_ghz = (
        1e-3 * predict_f01_mhz(params, np.array([0.31], dtype=np.float64))[0]
    )

    with pytest.warns(RuntimeWarning):
        result = correct_flux_from_f01(
            np.array([0.3, 0.3], dtype=np.float64),
            np.array([100.0, reachable_ghz], dtype=np.float64),
            params,
            max_abs_flux_correction=0.03,
        )

    np.testing.assert_allclose(result.corrected_fluxs, [0.3, 0.31], atol=1e-5)
    np.testing.assert_array_equal(result.accepted, [False, True])
    np.testing.assert_allclose(result.applied_flux_corrections, [0.0, 0.01], atol=1e-5)
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
