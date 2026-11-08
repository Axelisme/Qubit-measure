"""Cycle moments and joint fitting at the public analysis seam."""

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.readout_chain import CycleModel, CycleNuisances

from ._cycle_samples import cycle_samples


def example_nuisance() -> CycleNuisances:
    return CycleNuisances(
        np.array(
            [
                [[0.85, 0.10, 0.05], [0.20, 0.65, 0.15]],
                [[0.15, 0.80, 0.05], [0.70, 0.15, 0.15]],
                [[0.10, 0.20, 0.70], [0.20, 0.30, 0.50]],
            ]
        ),
        np.array([[-1.0, 0.0], [1.0, 0.2], [0.0, 1.8]]),
        0.25,
        np.array([0.7, 0.2, 0.1]),
        np.array([0.95, 0.80, 0.90]),
        np.array([0.7, 0.2, 0.1]),
    )


@pytest.mark.parametrize("axis", [np.array([]), np.ones((2, 1)), np.array([np.nan])])
def test_moments_reject_malformed_axis(axis: np.ndarray) -> None:
    model = CycleModel.identical(
        lambda x, p: np.exp(-x / p["tau"]), ("tau",), ((0.1, 10.0),)
    )
    with pytest.raises(ValueError, match="axis"):
        model.moments(axis, {"tau": 2.0}, example_nuisance())


@pytest.mark.parametrize(
    "change",
    [
        {"thermal": np.array([0.7, 0.2, 0.2])},
        {"variance": -1.0},
        {"centers": np.zeros((3, 3))},
        {"read_retention": np.array([0.8, -0.1, 0.9])},
    ],
)
def test_moments_reject_invalid_physical_nuisances(change: dict[str, object]) -> None:
    model = CycleModel.identical(
        lambda x, p: np.exp(-x / p["tau"]), ("tau",), ((0.1, 10.0),)
    )
    with pytest.raises(ValueError, match="nuisance|variance|centers|thermal|retention"):
        model.moments(
            np.array([0.1]), {"tau": 2.0}, replace(example_nuisance(), **change)
        )


def test_transition_tail_means_and_covariance_match_independent_cycles() -> None:
    model = CycleModel(
        lambda x, p: np.column_stack(
            [np.exp(-x / p["tau"]), np.exp(-x / p["tau"]), np.exp(-2 * x / p["tau"])]
        ),
        ("tau",),
        ((0.1, 10.0),),
    )
    nuisance = example_nuisance()
    moments = model.moments(np.array([0.7]), {"tau": 1.2}, nuisance)
    shots = cycle_samples(model, 0.7, {"tau": 1.2}, nuisance, 30000, 128)

    mean_error = np.sqrt(np.diag(moments.covariance[0]) / len(shots))
    assert np.max(np.abs(shots.mean(axis=0) - moments.means[0]) / mean_error) < 6
    centered = shots - moments.means[0]
    empirical = centered.T @ centered / len(shots)
    squared = centered**2
    covariance_error = np.sqrt(
        np.maximum(0.0, squared.T @ squared / len(shots) - empirical**2) / len(shots)
    )
    assert np.max(np.abs(empirical - moments.covariance[0]) / covariance_error) < 7


def coordinate_nuisance(q: np.ndarray) -> CycleNuisances:
    def probability(s: float, f: float) -> np.ndarray:
        return np.array([s * f, s * (1 - f), 1 - s])

    endpoints = np.array([probability(*pair) for pair in q[:12].reshape(6, 2)]).reshape(
        3, 2, 3
    )
    return CycleNuisances(
        endpoints,
        q[12:18].reshape(3, 2),
        q[18],
        probability(*q[19:21]),
        q[21:24],
        probability(*q[24:26]),
    )


def test_mean_jacobian_matches_numerical_differentiation_for_every_coordinate() -> None:
    model = CycleModel(
        lambda x, p: np.column_stack(
            [np.exp(-x / p["tau"]), np.exp(-x / p["tau"]), np.exp(-2 * x / p["tau"])]
        ),
        ("tau",),
        ((0.1, 10.0),),
    )
    # Six distinct simplex endpoints and nonzero read transitions exercise all derivatives.
    q = np.array(
        [
            0.9,
            0.8,
            0.8,
            0.2,
            0.9,
            0.2,
            0.8,
            0.7,
            0.3,
            0.4,
            0.5,
            0.6,
            -1.0,
            0.1,
            1.0,
            0.2,
            0.1,
            1.8,
            0.25,
            0.9,
            0.7,
            0.95,
            0.8,
            0.9,
            0.9,
            0.7,
        ]
    )
    values = np.r_[1.2, q]
    axis = np.array([0.7])
    reference = model.moments(axis, {"tau": values[0]}, coordinate_nuisance(values[1:]))
    for col, name in enumerate(reference.parameter_names):
        plus, minus = values.copy(), values.copy()
        plus[col] += 1e-5
        minus[col] -= 1e-5
        fp = model.moments(axis, {"tau": plus[0]}, coordinate_nuisance(plus[1:])).means
        fm = model.moments(
            axis, {"tau": minus[0]}, coordinate_nuisance(minus[1:])
        ).means
        np.testing.assert_allclose(
            reference.jacobian[:, :, col],
            (fp - fm) / 2e-5,
            rtol=1e-8,
            atol=1e-8,
            err_msg=name,
        )


def test_deterministic_read_jump_has_no_finite_wald_jacobian() -> None:
    nuisance = replace(
        example_nuisance(),
        read_retention=np.zeros(3),
        read_landing=np.array([1.0, 0.0, 0.0]),
    )
    model = CycleModel.identical(
        lambda x, p: np.exp(-x / p["tau"]), ("tau",), ((0.1, 10.0),)
    )
    with pytest.raises(ValueError, match="deterministic|regular"):
        model.moments(np.array([0.7]), {"tau": 1.2}, nuisance)


def test_model_rejects_unknown_frequency_coordinate() -> None:
    with pytest.raises(ValueError, match="frequency"):
        CycleModel(
            lambda x, p: np.zeros((len(x), 3)),
            ("tau",),
            ((0.1, 10.0),),
            oscillation_frequency="missing",
        )


def test_constant_cloud_has_gaussian_feature_moments() -> None:
    # All clouds coincide: the latent chain cannot alter independent Gaussian IQ.
    endpoints = np.repeat(np.eye(3)[:, None, :], 2, axis=1)
    nuisance = CycleNuisances(
        endpoints,
        np.tile([2.0, 3.0], (3, 1)),
        0.5,
        np.array([0.6, 0.3, 0.1]),
        np.ones(3),
        np.ones(3) / 3,
    )
    model = CycleModel.identical(
        lambda x, p: np.exp(-x / p["tau"]), ("tau",), ((0.1, 10.0),)
    )

    moments = model.moments(np.array([0.1, 1.0]), {"tau": 2.0}, nuisance)

    np.testing.assert_allclose(moments.means[:, :4], [[2, 3, 2, 3]] * 2)
    np.testing.assert_allclose(moments.means[:, 32:35], [[4.5, 6, 9.5]] * 2)
    np.testing.assert_allclose(moments.means[:, 35:38], [[31.5, 42, 66.5]] * 2)
    np.testing.assert_allclose(moments.covariance[:, 0, 0], 0.5)
    np.testing.assert_allclose(moments.covariance[:, 0, 2], 0, atol=1e-12)
    # Var(I²)=2*v²+4*mean(I)²*v, Cov(I,I²)=2*mean(I)*v.
    np.testing.assert_allclose(moments.covariance[:, 32, 32], 8.5)
    np.testing.assert_allclose(moments.covariance[:, 0, 32], 2)
    assert moments.jacobian.shape == (2, 65, len(moments.parameter_names))
    np.testing.assert_allclose(moments.jacobian[:, :, 0], 0, atol=1e-12)
