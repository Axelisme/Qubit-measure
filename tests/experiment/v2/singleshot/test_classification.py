from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from zcu_tools.experiment.v2.singleshot.ge import optimize_ge_radius
from zcu_tools.experiment.v2.singleshot.util import (
    classify_result,
    plot_with_classified,
)
from zcu_tools.program.base.improve_acquire import SingleShotMixin
from zcu_tools.utils.fitting.singleshot import transition_state_circle_probabilities
from zcu_tools.utils.shot_classification import gaussian_region_probability


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("remove_offset", [False, True])
def test_program_matches_offline(radius: float, remove_offset: bool) -> None:
    signals = np.array([[-1, 1, 0, 0.2, -0.2, 4, -1 + radius]], dtype=np.complex128)
    expected = classify_result(signals, -1, 1, radius)
    assert np.all(sum(mask.astype(int) for mask in expected) == 1)
    assert expected[2][0, 2]  # Exact tie.
    offset = 0.25 if remove_offset else 0.0
    raw = (np.stack((signals.real, signals.imag), axis=-1) + offset) * 10
    program = SimpleNamespace(
        ro_chs={0: {"length": 10}}, soccfg={"readouts": [{"iq_offset": offset}]}
    )
    actual = SingleShotMixin._apply_classification(
        cast(SingleShotMixin, program), [raw], -1, 1, radius, remove_offset
    )
    np.testing.assert_array_equal(actual[0], np.stack(expected[:2], axis=-1))
    assert actual[0].dtype == np.float64


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


def test_ge_optimizer_can_exceed_half_separation() -> None:
    # Both prepared clouds lie outside their old radius limit, but on their own side.
    g = np.full(100, -1 + 1.4j)
    e = np.full(100, 1 + 1.4j)
    radius = optimize_ge_radius(g, e, -1, 1, np.eye(3)[:2], 0.3, consider_other=False)
    assert 1.4 < radius <= 2


def test_plot_boundaries_stay_in_assigned_half_planes() -> None:
    fig, ax = plt.subplots()
    try:
        plot_with_classified(ax, np.array([-1, 1, 0j]), -1, 1, 3)
        assert len(ax.patches) == 2
        assert np.asarray(ax.patches[0].get_path().vertices)[:, 0].max() <= 1e-12
        assert np.asarray(ax.patches[1].get_path().vertices)[:, 0].min() >= -1e-12
    finally:
        plt.close(fig)


def test_transition_regions_match_sampled_transition_distribution() -> None:
    from scipy.integrate import cumulative_trapezoid
    from zcu_tools.utils.fitting.singleshot import calc_fc

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


def test_rabi_optimizer_searches_up_to_separation() -> None:
    from zcu_tools.experiment.v2.singleshot.rabi_fit import (
        RabiPhysicalParams,
        _confusion_matrix,
    )

    params = RabiPhysicalParams(0.1, 0.5, -1, 1, 0.8, 0.3, 0.2, None, 1.0)
    radius, matrix, condition = _confusion_matrix(params)
    assert 1 < radius <= 2
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    assert np.isfinite(condition)
