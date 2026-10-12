"""Public IQ classification and population probability contracts."""

import numpy as np
import pytest
from zcu_tools.analysis.fitting.singleshot import transition_state_circle_probabilities
from zcu_tools.analysis.singleshot import (
    calc_populations,
    classify_rabi_iq,
    classify_result,
    correct_populations,
    raw_population_signal,
)
from zcu_tools.utils.shot_classification import gaussian_region_probability


def test_population_conversion_and_readout_correction() -> None:
    ge = np.array([[0.3, 0.5], [0.4, 0.4]], dtype=np.float64)
    populations = calc_populations(ge)
    np.testing.assert_allclose(populations, [[0.3, 0.5, 0.2], [0.4, 0.4, 0.2]])
    matrix = np.array([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
    measured = populations @ matrix
    np.testing.assert_allclose(correct_populations(measured, matrix), populations)
    assert correct_populations(populations, None) is populations
    np.testing.assert_array_equal(raw_population_signal([ge[None, ...]]), ge)
    with pytest.raises(np.linalg.LinAlgError):
        correct_populations(populations, np.zeros((3, 3)))


def test_rabi_classification_reports_per_coordinate_fractions() -> None:
    signals = np.array([[-1, 1, 0], [-1, -1, 1]], dtype=np.complex128)
    np.testing.assert_allclose(
        classify_rabi_iq(signals, -1, 1, 0.5), [[1 / 3, 1 / 3], [2 / 3, 1 / 3]]
    )
    with pytest.raises(ValueError, match="shape"):
        classify_rabi_iq(signals[0], -1, 1, 0.5)


def test_boundaries_and_degenerate_centers() -> None:
    g, e, other = classify_result(np.array([-2, 0, 2], dtype=np.complex128), -1, 1, 1)
    assert other.all() and not g.any() and not e.any()
    with pytest.raises(ValueError, match="distinct"):
        classify_result(np.array([0j]), 0j, 0j, 1)


@pytest.mark.parametrize("radius", [0.7, 1.5, 2.0, 3.0])
def test_gaussian_regions_match_raw_shots(radius: float) -> None:
    rng = np.random.default_rng(42)
    sigma = 0.8
    shots = -1 + sigma * (rng.normal(size=250000) + 1j * rng.normal(size=250000))
    measured = [mask.mean() for mask in classify_result(shots, -1, 1, radius)]
    ground, excited = transition_state_circle_probabilities(
        -1, 1, sigma, 0.3, 0, radius
    )
    np.testing.assert_allclose(ground, measured, atol=0.004)
    np.testing.assert_allclose(excited, ground[[1, 0, 2]], atol=1e-10)
    assert np.all(ground >= 0)
    assert ground.sum() == pytest.approx(1)


def test_midpoint_other_mass_and_large_radius_limit() -> None:
    mass = gaussian_region_probability(np.array([1.0]), 0.5, 20, 2)
    assert mass[0] == pytest.approx(0.5, abs=1e-9)
    ground, _ = transition_state_circle_probabilities(-1, 1, 0.3, 0.2, 0.8, 2)
    assert ground.sum() == pytest.approx(1)
    assert np.all(ground >= 0)


def test_transition_regions_match_sampled_transition_distribution() -> None:
    from scipy.integrate import cumulative_trapezoid
    from zcu_tools.analysis.fitting.singleshot import calc_fc

    rng = np.random.default_rng(13)
    p_avg, length_ratio, sigma, radius = 0.3, 1.2, 0.6, 2.0
    grid = np.linspace(0, 1, 10001)
    rows = transition_state_circle_probabilities(
        -1, 1, sigma, p_avg, length_ratio, radius
    )
    for index, (start, direction, rate, back) in enumerate(
        [
            (-1, 2, p_avg * length_ratio, (1 - p_avg) * length_ratio),
            (1, -2, (1 - p_avg) * length_ratio, p_avg * length_ratio),
        ]
    ):
        density = calc_fc(grid, rate, back)
        cdf = cumulative_trapezoid(density, grid, initial=0)
        cdf /= cdf[-1]
        fractions = np.interp(rng.random(250000), cdf, grid)
        fractions[rng.random(250000) < np.exp(-rate)] = 0
        shots = (
            start
            + direction * fractions
            + sigma * (rng.normal(size=250000) + 1j * rng.normal(size=250000))
        )
        empirical = [mask.mean() for mask in classify_result(shots, -1, 1, radius)]
        np.testing.assert_allclose(rows[index], empirical, atol=0.004)
