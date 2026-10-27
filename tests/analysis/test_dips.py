"""Public magnitude dip selection contracts on synthetic spectra."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.dips import DipFinderOptions, find_spectrum_dips


def test_narrow_dip_keeps_full_depth_and_outranks_deeper_ripple() -> None:
    freqs = np.arange(201, dtype=np.float64)
    magnitude = np.full_like(freqs, 20.0)
    magnitude[20:81] -= 8 * (1 - np.abs(freqs[20:81] - 50) / 30)
    magnitude[150] -= 4
    dips = find_spectrum_dips(freqs, magnitude, DipFinderOptions(max_fwhm=100))
    assert [dip.freq for dip in dips] == [150.0, 50.0]
    assert dips[0].depth == pytest.approx(4.0)
    assert dips[0].fwhm == pytest.approx(1.0)
    assert dips[0].score == pytest.approx(4.0)
    assert dips[1].fwhm == pytest.approx(30.0)
    assert dips[1].score == pytest.approx(8 / 30)


def test_fwhm_limit_excludes_wide_dips_and_can_be_relaxed() -> None:
    freqs = np.arange(201, dtype=np.float64)
    magnitude = 20 - 8 * np.maximum(0, 1 - np.abs(freqs - 100) / 30)
    assert find_spectrum_dips(freqs, magnitude) == ()
    dips = find_spectrum_dips(freqs, magnitude, DipFinderOptions(max_fwhm=30))
    assert [dip.freq for dip in dips] == [100.0]
    assert dips[0].fwhm == pytest.approx(30.0)


def test_count_and_spacing_select_by_score_in_actual_mhz() -> None:
    freqs = np.arange(101, dtype=np.float64) * 2
    magnitude = np.full_like(freqs, 20.0)
    magnitude[[20, 23, 60]] -= [3, 4, 2]
    count_limited = find_spectrum_dips(freqs, magnitude, DipFinderOptions(max_dips=2))
    assert [dip.freq for dip in count_limited] == [46.0, 40.0]
    spaced = find_spectrum_dips(
        freqs, magnitude, DipFinderOptions(min_separation=7, max_dips=2)
    )
    assert [dip.freq for dip in spaced] == [46.0, 120.0]
    at_boundary = find_spectrum_dips(
        freqs, magnitude, DipFinderOptions(min_separation=6)
    )
    assert [dip.freq for dip in at_boundary] == [46.0, 40.0, 120.0]


def test_two_sample_dip_survives_broad_oscillating_background() -> None:
    freqs = np.arange(201, dtype=np.float64)
    magnitude = 20 - np.cos((freqs - 70.5) * np.pi / 50)
    magnitude[70:72] -= 4
    dips = find_spectrum_dips(freqs, magnitude)
    assert len(dips) == 1
    assert dips[0].freq in (70, 71)
    assert dips[0].depth >= 4
    assert 2 <= dips[0].fwhm < 4


def test_optional_smoothing_reduces_single_sample_depth() -> None:
    freqs = np.arange(101, dtype=np.float64)
    magnitude = np.full_like(freqs, 20.0)
    magnitude[50] -= 4
    dips = find_spectrum_dips(freqs, magnitude, DipFinderOptions(smoothing_sigma=1))
    assert len(dips) == 1
    assert dips[0].freq == 50
    assert 1 < dips[0].depth < 2
    assert dips[0].fwhm > 2


def test_prominence_factor_controls_noise_rejection() -> None:
    freqs = np.arange(101, dtype=np.float64)
    magnitude = 20 + np.sin(freqs * np.pi / 2)
    magnitude[50] -= 4
    permissive = find_spectrum_dips(
        freqs, magnitude, DipFinderOptions(prominence_factor=0)
    )
    strict = find_spectrum_dips(freqs, magnitude, DipFinderOptions(prominence_factor=5))
    assert len(permissive) > 1
    assert strict == ()


@pytest.mark.parametrize(
    "build_options",
    [
        lambda: DipFinderOptions(smoothing_sigma=-1.0),
        lambda: DipFinderOptions(prominence_factor=float("nan")),
        lambda: DipFinderOptions(max_fwhm=0.0),
        lambda: DipFinderOptions(min_separation=float("inf")),
        lambda: DipFinderOptions(min_separation=True),
        lambda: DipFinderOptions(max_dips=0),
        lambda: replace(DipFinderOptions(), max_dips=1.5),
        lambda: DipFinderOptions(max_dips=True),
    ],
)
def test_invalid_controls_raise_value_error(
    build_options: Callable[[], DipFinderOptions],
) -> None:
    with pytest.raises(ValueError, match="must be"):
        build_options()


@pytest.mark.parametrize("defect", ["short", "shape", "nan", "duplicate", "negative"])
def test_invalid_spectra_raise_value_error(defect: str) -> None:
    freqs = np.arange(101, dtype=np.float64)
    magnitude = np.ones_like(freqs)
    if defect == "short":
        freqs, magnitude = freqs[:10], magnitude[:10]
    elif defect == "shape":
        magnitude = magnitude[:, None]
    elif defect == "nan":
        magnitude[50] = np.nan
    elif defect == "duplicate":
        freqs[50] = freqs[51]
    else:
        magnitude[50] = -1
    with pytest.raises(ValueError, match="needs|nonnegative"):
        find_spectrum_dips(freqs, magnitude)


def test_sorted_scan_invariance_and_local_step_floor() -> None:
    freqs = np.arange(101, dtype=np.float64) * 2
    freqs[51:] += 2
    magnitude = 30 + 0.1 * freqs
    magnitude[50] -= 4
    before = magnitude.copy()
    forward = find_spectrum_dips(freqs, magnitude)
    backward = find_spectrum_dips(freqs[::-1], magnitude[::-1])
    assert forward == backward
    assert forward[0].freq == 100
    assert forward[0].depth == pytest.approx(4)
    assert forward[0].fwhm == pytest.approx(3)
    np.testing.assert_array_equal(magnitude, before)
