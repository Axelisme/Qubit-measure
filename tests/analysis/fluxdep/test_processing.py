from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep import (
    cast2real_and_norm,
    contrast_limits,
    diff_mirror,
    downsample_points,
    onetone_background,
    spectrum2d_findpoint,
)


@pytest.mark.parametrize(
    "invalid_call", [cast2real_and_norm], ids=["cast2real_and_norm"]
)
def test_cast2real_and_norm_rejects_positional_phase_option(
    invalid_call: Callable[..., object],
) -> None:
    signals = np.ones((3, 5), dtype=np.complex128)
    with pytest.raises(TypeError, match="positional"):
        invalid_call(signals, False, 1, "gaussian")


def test_cast2real_and_norm_keeps_constant_rows_finite() -> None:
    signals = np.ones((3, 5), dtype=np.complex128)
    real = cast2real_and_norm(signals)
    assert real.shape == signals.shape
    assert np.isfinite(real).all()


def test_spectrum2d_findpoint_limits_to_three_peaks_per_row() -> None:
    devs = np.array([0.0, 1.0], dtype=np.float64)
    freqs = np.arange(20, dtype=np.float64)
    real = np.zeros((2, 20), dtype=np.float64)
    real[0, [2, 7, 12, 17]] = [2.0, 3.0, 4.0, 5.0]
    real[1, [4, 10]] = [3.0, 4.0]
    s_dev, s_freq = spectrum2d_findpoint(devs, freqs, real, threshold=1.0)
    assert s_dev.shape == s_freq.shape
    assert int((s_dev == 0.0).sum()) == 3
    assert int((s_dev == 1.0).sum()) == 2


def test_downsample_points_is_deterministic() -> None:
    xs = np.array([0.0, 0.01, 0.5, 0.5], dtype=np.float64)
    ys = np.array([0.0, 0.01, 0.5, 0.6], dtype=np.float64)
    first = downsample_points(xs, ys, threshold=0.1)
    second = downsample_points(xs, ys, threshold=0.1)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.bool_


def test_diff_mirror_rejects_axis_shape_mismatch() -> None:
    xs = np.linspace(0.0, 1.0, 4).astype(np.float64)
    data = np.ones((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="first dimension"):
        diff_mirror(xs, data, center=0.5)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.full(4, np.nan), (0.0, 1.0)),
        (np.zeros(50), (0.0, 1.0)),
        (np.full(50, -2.0), (-2.0, -1.0)),
        (np.array([-3.0, *np.zeros(98), 0.0]), (-3.0, 0.0)),
    ],
    ids=["no finite", "all zero", "constant", "outlier below zero max"],
)
def test_contrast_limits_always_span_a_positive_range(values, expected) -> None:
    assert contrast_limits(values.astype(np.float64)) == pytest.approx(expected)


def test_onetone_background_puts_the_dip_at_the_top_of_its_limits() -> None:
    devs, freqs = 20, 30
    center = np.linspace(8, 22, devs).astype(int)
    magnitude = np.ones((devs, freqs))
    magnitude[np.arange(devs), center] = 0.2
    background = onetone_background(magnitude.astype(np.complex128))
    assert background.values.shape == (devs, freqs)
    assert background.vmin < background.vmax
    # Colour level in [0, 1] under the returned limits; gray_r draws 1 darkest.
    span = background.vmax - background.vmin
    level = np.clip((background.values - background.vmin) / span, 0.0, 1.0)
    assert np.all(level[np.arange(devs), center] > 0.9)
    assert np.median(level) < 0.1
