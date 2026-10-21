"""Sampling density of supplied fringe frequencies at the public fitting seam."""

import numpy as np
import pytest
from zcu_tools.analysis import fitting


@pytest.mark.parametrize(
    ("times", "frequency", "expected", "gap", "samples", "status"),
    [
        ([0.0, 0.25, 0.5, 0.75], 1.0, None, 0.25, 4.0, "adequate"),
        ([0.0, 0.5, 1.0], 0.4, None, 0.5, 5.0, "adequate"),
        ([0.0, 1.0, 2.0], 0.4, None, 1.0, 2.5, "undersampled"),
        ([0.0, 1.0, 2.0], 0.1, 0.9, 1.0, 1.1111111111111112, "undersampled"),
        ([0.0, 0.5, 1.0], -0.4, -0.1, 0.5, 5.0, "adequate"),
        ([0.0, 0.5, 1.0], 0.0, -0.4, 0.5, 5.0, "adequate"),
        ([0.0, 0.25, 1.25], 0.5, None, 1.0, 2.0, "undersampled"),
        ([1.25, 0.25, 0.25, 0.0], 0.5, None, 1.0, 2.0, "undersampled"),
    ],
    ids=[
        "four-samples-boundary",
        "five-samples",
        "below-four",
        "expected-rate-exposes-alias-risk",
        "absolute-rate-and-fit-lower-bound",
        "zero-fit-uses-expected",
        "largest-gap-not-average",
        "sort-and-deduplicate",
    ],
)
def test_sampling_density_uses_largest_gap_and_known_frequency_lower_bound(
    times: list[float],
    frequency: float,
    expected: float | None,
    gap: float,
    samples: float,
    status: str,
) -> None:
    assert hasattr(fitting, "assess_fringe_sampling"), "Missing sampling assessment"
    coordinates = np.array(times)
    before = coordinates.copy()
    quality = fitting.assess_fringe_sampling(coordinates, frequency, expected)
    assert isinstance(quality, fitting.FringeSamplingQuality)
    assert quality.max_gap_us == pytest.approx(gap)
    assert quality.samples_per_cycle == pytest.approx(samples)
    assert quality.status == status
    assert quality.reason is None
    np.testing.assert_array_equal(coordinates, before)


@pytest.mark.parametrize(
    ("times", "frequency", "expected", "gap", "reason"),
    [
        ([], 1.0, None, None, "no_gap"),
        ([1.0], 1.0, None, None, "no_gap"),
        ([1.0, 1.0], 1.0, None, None, "no_gap"),
        ([0.0, 0.5, 1.0], 0.0, None, 0.5, "zero_frequency"),
        ([0.0, 0.5, 1.0], 0.0, 0.0, 0.5, "zero_frequency"),
    ],
)
def test_unestimable_density_retains_gap_and_explains_unavailability(
    times: list[float],
    frequency: float,
    expected: float | None,
    gap: float | None,
    reason: str,
) -> None:
    assert hasattr(fitting, "assess_fringe_sampling"), "Missing sampling assessment"
    quality = fitting.assess_fringe_sampling(np.array(times), frequency, expected)
    assert quality.status == "unavailable"
    assert quality.max_gap_us == gap
    assert quality.samples_per_cycle is None
    assert quality.reason == reason


@pytest.mark.parametrize(
    ("times", "frequency", "expected"),
    [
        (np.array(1.0), 1.0, None),
        (np.array([[0.0, 1.0]]), 1.0, None),
        (np.array([0.0, np.nan]), 1.0, None),
        (np.array([0.0, np.inf]), 1.0, None),
        (np.array([0.0, 1.0]), np.nan, None),
        (np.array([0.0, 1.0]), np.inf, None),
        (np.array([0.0, 1.0]), 1.0, np.nan),
        (np.array([0.0, 1.0]), 1.0, np.inf),
    ],
)
def test_invalid_coordinates_or_frequencies_raise_at_entry(
    times: np.ndarray, frequency: float, expected: float | None
) -> None:
    assert hasattr(fitting, "assess_fringe_sampling"), "Missing sampling assessment"
    with pytest.raises(ValueError):
        fitting.assess_fringe_sampling(times, frequency, expected)
