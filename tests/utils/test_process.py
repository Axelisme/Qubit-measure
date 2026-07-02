"""
Unit tests for lib/zcu_tools/utils/process.py

All expected values are derived from the CURRENT implementation, pinning the
existing semantics as a regression anchor.  peak_n_avg tests are intentionally
written against the old loop-based behaviour so that the argpartition
optimisation in step 2 cannot silently change results.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from zcu_tools.utils.process import (
    find_rotate_angle,
    minus_background,
    minus_mean,
    minus_median,
    peak_n_avg,
    rotate2real,
    rotate_phase,
    smooth_signal1d,
    smooth_signal_nd,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_complex(rng: np.random.Generator, n: int = 50) -> np.ndarray:
    """Return a 1-D complex128 array with a known dominant direction."""
    angle = np.pi / 6  # 30 degrees
    spread = rng.standard_normal(n)
    noise = rng.standard_normal((n, 2)) * 0.05
    real = spread * np.cos(angle) + noise[:, 0]
    imag = spread * np.sin(angle) + noise[:, 1]
    return (real + 1j * imag).astype(np.complex128)


# ===========================================================================
# rotate2real
# ===========================================================================


class TestRotate2Real:
    def test_output_variance_on_real_axis(self) -> None:
        """After rotation the imaginary variance should be minimal."""
        signals = _make_complex(RNG)
        rotated = rotate2real(signals)
        assert rotated.dtype == np.complex128
        # real variance >> imag variance after rotation
        assert np.var(rotated.real) > 10 * np.var(rotated.imag)

    def test_amplitude_preserved(self) -> None:
        """rotate2real must not change signal amplitude."""
        signals = _make_complex(RNG)
        assert_allclose(np.abs(rotate2real(signals)), np.abs(signals), rtol=1e-12)

    def test_2d_input_accepted(self) -> None:
        """rotate2real should handle 2-D input (internally flattened)."""
        signals = _make_complex(RNG, 40).reshape(8, 5)
        out = rotate2real(signals)
        assert out.shape == signals.shape

    def test_too_few_valid_returns_input(self) -> None:
        """With fewer than 2 non-NaN values the original array is returned."""
        signals = np.array([1.0 + 1j, np.nan + 0j], dtype=np.complex128)
        out = rotate2real(signals)
        assert out is signals or np.array_equal(out, signals, equal_nan=True)

    def test_all_nan_returns_input(self) -> None:
        signals = np.array([np.nan + 0j, np.nan + 0j], dtype=np.complex128)
        out = rotate2real(signals)
        assert np.all(np.isnan(out))


# ===========================================================================
# rotate_phase
# ===========================================================================


class TestRotatePhase:
    def test_zero_slope_is_identity(self) -> None:
        freqs = np.linspace(4.0, 5.0, 10)
        signals = _make_complex(RNG, 10)
        out = rotate_phase(freqs, signals, phase_slope=0.0)
        assert_allclose(out, signals, atol=1e-14)

    def test_known_90_degree_rotation(self) -> None:
        """At freq=0.5, slope=360 -> angle = 0.5*360*(pi/180) = pi -> rotate by pi."""
        freqs = np.array([0.5])
        # signal = 1+0j, rotate by pi -> -1+0j (approximately)
        signals = np.array([1.0 + 0j], dtype=np.complex128)
        out = rotate_phase(freqs, signals, phase_slope=360.0)
        assert_allclose(out.real, [-1.0], atol=1e-14)
        assert_allclose(out.imag, [0.0], atol=1e-14)

    def test_output_shape_preserved(self) -> None:
        freqs = np.linspace(1.0, 2.0, 20)
        signals = _make_complex(RNG, 20)
        out = rotate_phase(freqs, signals, phase_slope=5.0)
        assert out.shape == signals.shape

    def test_amplitude_preserved(self) -> None:
        """Phase rotation is unitary; amplitude must not change."""
        freqs = np.linspace(4.0, 6.0, 30)
        signals = _make_complex(RNG, 30)
        out = rotate_phase(freqs, signals, phase_slope=12.3)
        assert_allclose(np.abs(out), np.abs(signals), rtol=1e-12)


# ===========================================================================
# minus_background / minus_median / minus_mean
# ===========================================================================


class TestMinusBackground:
    def test_dispatches_median(self) -> None:
        a = RNG.random((4, 4))
        assert_allclose(minus_background(a, method="median"), minus_median(a))

    def test_dispatches_mean(self) -> None:
        a = RNG.random((4, 4))
        assert_allclose(minus_background(a, method="mean"), minus_mean(a))

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid method"):
            minus_background(np.array([1.0, 2.0]), method="rms")


class TestMinusMedian:
    def test_1d_real_no_axis(self) -> None:
        a = np.array([1.0, 3.0, 5.0, 7.0])
        out = minus_median(a)
        assert_allclose(out, a - np.median(a))

    def test_does_not_modify_input(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        a_copy = a.copy()
        minus_median(a)
        np.testing.assert_array_equal(a, a_copy)

    def test_2d_axis0(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = minus_median(a, axis=0)
        # axis=0: subtract per-column median
        col_med = np.median(a, axis=0)
        assert_allclose(out, a - col_med)

    def test_2d_axis1(self) -> None:
        a = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        out = minus_median(a, axis=1)
        row_med = np.median(a, axis=1, keepdims=True)
        assert_allclose(out, a - row_med)

    def test_nan_ignored(self) -> None:
        a = np.array([1.0, np.nan, 3.0, 5.0])
        out = minus_median(a)
        expected = a - np.nanmedian(a)
        assert_allclose(out[~np.isnan(out)], expected[~np.isnan(expected)])

    def test_all_nan_returns_all_nan(self) -> None:
        a = np.array([np.nan, np.nan])
        out = minus_median(a)
        assert np.all(np.isnan(out))

    def test_complex_no_axis(self) -> None:
        a = np.array([1.0 + 2j, 3.0 + 4j, 5.0 + 6j], dtype=np.complex128)
        out = minus_median(a)
        assert_allclose(out.real, a.real - np.median(a.real))
        assert_allclose(out.imag, a.imag - np.median(a.imag))

    def test_invalid_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid axis"):
            minus_median(np.array([[1.0, 2.0]]), axis=[0, 1])  # type: ignore[arg-type]


class TestMinusMean:
    def test_1d_real_no_axis(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        out = minus_mean(a)
        assert_allclose(out, a - np.mean(a))

    def test_does_not_modify_input(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        a_copy = a.copy()
        minus_mean(a)
        np.testing.assert_array_equal(a, a_copy)

    def test_2d_axis0(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = minus_mean(a, axis=0)
        col_mean = np.mean(a, axis=0)
        assert_allclose(out, a - col_mean)

    def test_nan_ignored(self) -> None:
        a = np.array([1.0, np.nan, 3.0])
        out = minus_mean(a)
        expected = a - np.nanmean(a)
        mask = ~np.isnan(out)
        assert_allclose(out[mask], expected[mask])

    def test_all_nan_returns_all_nan(self) -> None:
        a = np.array([np.nan, np.nan])
        out = minus_mean(a)
        assert np.all(np.isnan(out))

    def test_invalid_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid axis"):
            minus_mean(np.array([[1.0, 2.0]]), axis="bad")  # type: ignore[arg-type]


# ===========================================================================
# find_rotate_angle
# ===========================================================================


class TestFindRotateAngle:
    def test_returns_float(self) -> None:
        signals = _make_complex(RNG)
        angle = find_rotate_angle(signals)
        assert isinstance(angle, float)

    def test_angle_in_range(self) -> None:
        """Returned angle must lie in (-pi/2, pi/2] — eigenvector convention."""
        for seed in range(5):
            rng2 = np.random.default_rng(seed)
            signals = _make_complex(rng2)
            angle = find_rotate_angle(signals)
            assert -np.pi / 2 <= angle <= np.pi / 2

    def test_wrong_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="complex"):
            find_rotate_angle(np.array([1.0, 2.0]))  # float64, not complex128

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            find_rotate_angle(np.array([1.0 + 1j], dtype=np.complex128))

    def test_2d_input_accepted(self) -> None:
        signals = _make_complex(RNG, 20).reshape(4, 5)
        angle = find_rotate_angle(signals)
        assert isinstance(angle, float)

    def test_known_angle(self) -> None:
        """Pure real signal -> dominant direction is 0 -> angle near 0."""
        rng2 = np.random.default_rng(7)
        x = rng2.standard_normal(200)
        signals = (x + 0j).astype(np.complex128)
        angle = find_rotate_angle(signals)
        assert abs(angle) < 0.1  # within ~6 degrees


# ===========================================================================
# peak_n_avg — comprehensive semantic lock-down
# ===========================================================================


class TestPeakNAvgMaxMode:
    """Tests for peak_n_avg with mode='max' (default)."""

    def test_basic_top1(self) -> None:
        """n=1 returns the single maximum."""
        data = np.array([1.0, 5.0, 3.0, 2.0])
        assert peak_n_avg(data, n=1) == pytest.approx(5.0)

    def test_basic_top2(self) -> None:
        """n=2 returns the average of the two largest elements."""
        data = np.array([1.0, 5.0, 3.0, 2.0])
        # top-2: 5 and 3 -> avg = 4.0
        assert peak_n_avg(data, n=2) == pytest.approx(4.0)

    def test_basic_top3(self) -> None:
        data = np.array([1.0, 5.0, 3.0, 2.0])
        # top-3: 5, 3, 2 -> avg = 10/3
        assert peak_n_avg(data, n=3) == pytest.approx(10.0 / 3)

    def test_n_equals_size_returns_nanmean(self) -> None:
        """n >= valid count falls back to nanmean (condition: sum <= n)."""
        data = np.array([1.0, 5.0, 3.0, 2.0])
        # n_valid=4, n=4 -> sum(~nan)=4 <= 4 -> nanmean = 2.75
        assert peak_n_avg(data, n=4) == pytest.approx(2.75)

    def test_n_greater_than_size_returns_nanmean(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        assert peak_n_avg(data, n=10) == pytest.approx(np.nanmean(data))

    def test_nan_inputs_skipped(self) -> None:
        """NaN values are excluded from top-n selection."""
        data = np.array([1.0, np.nan, 5.0, 3.0, np.nan, 2.0])
        # n_valid=4, n=2 -> top-2 of {1,5,3,2} = 5 and 3 -> avg=4.0
        assert peak_n_avg(data, n=2) == pytest.approx(4.0)

    def test_nan_count_edge_exact(self) -> None:
        """When n_valid == n + 1 the loop path (not nanmean) runs."""
        data = np.array([np.nan, 4.0, 2.0])
        # n_valid=2, n=1 -> sum(~nan)=2 > 1 -> loop; top-1=4.0
        assert peak_n_avg(data, n=1) == pytest.approx(4.0)

    def test_n_valid_equals_n_fallback(self) -> None:
        """n_valid == n -> condition sum(~nan) <= n -> nanmean fallback."""
        data = np.array([np.nan, 4.0, 2.0])
        # n_valid=2, n=2 -> 2 <= 2 -> nanmean=3.0
        assert peak_n_avg(data, n=2) == pytest.approx(3.0)

    def test_ties_both_selected(self) -> None:
        """Tied largest values: both are picked when n=2."""
        data = np.array([5.0, 3.0, 5.0, 1.0])
        # top-2: both 5.0s -> avg=5.0
        assert peak_n_avg(data, n=2) == pytest.approx(5.0)

    def test_2d_input_flattened(self) -> None:
        """peak_n_avg must handle 2-D input (internally flattened)."""
        data = np.array([[1.0, 5.0], [3.0, 2.0]])
        assert peak_n_avg(data, n=2) == pytest.approx(4.0)

    def test_n_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n should be positive"):
            peak_n_avg(np.array([1.0, 2.0]), n=0)

    def test_n_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n should be positive"):
            peak_n_avg(np.array([1.0, 2.0]), n=-1)

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            peak_n_avg(np.array([1.0, 2.0]), n=1, mode="median")  # type: ignore[arg-type]

    def test_large_random_consistency(self) -> None:
        """Optimised result must match the old loop on a large array."""
        rng2 = np.random.default_rng(99)
        data = rng2.random(10_000)

        # Reference: old loop implementation (pinned here to detect drift)
        def _old(d: np.ndarray, n: int) -> float:
            if np.sum(~np.isnan(d)) <= n:
                return float(np.nanmean(d))
            _d = d.copy().flatten()
            peaks = np.empty(n, dtype=d.dtype)
            for i in range(n):
                idx = int(np.nanargmax(_d))
                peaks[i], _d[idx] = _d[idx], np.nan
            return float(np.mean(peaks))

        for n in [1, 5, 20, 50]:
            assert peak_n_avg(data, n=n) == pytest.approx(_old(data, n), rel=1e-12)


class TestPeakNAvgMinMode:
    def test_basic_min_top1(self) -> None:
        data = np.array([1.0, 5.0, 3.0, 2.0])
        assert peak_n_avg(data, n=1, mode="min") == pytest.approx(1.0)

    def test_basic_min_top2(self) -> None:
        data = np.array([1.0, 5.0, 3.0, 2.0])
        # bottom-2: 1, 2 -> avg=1.5
        assert peak_n_avg(data, n=2, mode="min") == pytest.approx(1.5)

    def test_min_nan_skipped(self) -> None:
        data = np.array([np.nan, 1.0, 5.0, 3.0, np.nan, 2.0])
        # bottom-2 of {1,5,3,2} = 1 and 2 -> avg=1.5
        assert peak_n_avg(data, n=2, mode="min") == pytest.approx(1.5)

    def test_min_n_greater_than_valid_returns_nanmean(self) -> None:
        data = np.array([1.0, 2.0])
        assert peak_n_avg(data, n=5, mode="min") == pytest.approx(1.5)

    def test_large_random_consistency_min(self) -> None:
        rng2 = np.random.default_rng(123)
        data = rng2.random(5_000)

        def _old_min(d: np.ndarray, n: int) -> float:
            if np.sum(~np.isnan(d)) <= n:
                return float(np.nanmean(d))
            _d = d.copy().flatten()
            peaks = np.empty(n, dtype=d.dtype)
            for i in range(n):
                idx = int(np.nanargmin(_d))
                peaks[i], _d[idx] = _d[idx], np.nan
            return float(np.mean(peaks))

        for n in [1, 10, 100]:
            assert peak_n_avg(data, n=n, mode="min") == pytest.approx(
                _old_min(data, n), rel=1e-12
            )


class TestWaveletSmoothing:
    def test_wavelet_preserves_shape_and_nan_mask(self) -> None:
        x = np.linspace(0.0, 2.0 * np.pi, 64)
        data = np.vstack([np.sin(x), np.cos(x)]).astype(np.float64)
        data[1, 7] = np.nan

        out = smooth_signal1d(data, method="wavelet", sigma=1.0, axis=1)

        assert out.shape == data.shape
        assert np.isnan(out[1, 7])
        assert np.all(np.isfinite(out[~np.isnan(data)]))

    def test_wavelet_handles_complex_nd_axis(self) -> None:
        x = np.linspace(0.0, 2.0 * np.pi, 32)
        real = np.vstack([np.sin(x), np.cos(x)])
        data = (real + 0.25j * real).astype(np.complex128)

        out = smooth_signal_nd(data, method="wavelet", sigma=1.0, axes=(1,))

        assert out.shape == data.shape
        assert np.iscomplexobj(out)
        assert np.all(np.isfinite(np.real(out)))
        assert np.all(np.isfinite(np.imag(out)))

    def test_gaussian_fallback_matches_scipy(self) -> None:
        from scipy.ndimage import gaussian_filter1d

        data = RNG.random((4, 20))
        out = smooth_signal1d(data, method="gaussian", sigma=1.5, axis=1)
        expected = gaussian_filter1d(data, sigma=1.5, axis=1)

        assert_allclose(out, expected)

    def test_invalid_smooth_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid smoothing method"):
            smooth_signal1d(np.arange(8.0), method="median")  # type: ignore[arg-type]
