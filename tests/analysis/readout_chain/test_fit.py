"""Saved-summary joint fitting of three-state cycles."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from zcu_tools.analysis.readout_chain import (
    CycleFit,
    CycleModel,
    CycleNuisances,
    CycleSums,
    fit_cycle,
)

from ._cycle_samples import sampled_sums
from ._exact_moments import curve_moments, repeated_curve


def expect_curve_nuisances(fit: CycleFit) -> CycleNuisances:
    if not isinstance(fit.nuisances, CycleNuisances):
        raise AssertionError("curve model must return curve nuisances")
    return fit.nuisances


def planted_model(kind: str = "t1"):
    if kind == "t1":
        model = repeated_curve(
            lambda x, p: np.exp(-x / p["tau"]),
            ("tau",),
            ((0.2, 5.0),),
            derivative=lambda x, p: (x * np.exp(-x / p["tau"]) / p["tau"] ** 2)[
                :, None
            ],
        )
    else:
        model = repeated_curve(
            lambda x, p: np.sin(np.pi * p["frequency"] * x) ** 2,
            ("frequency",),
            ((0.1, 2.0),),
            oscillation_frequency="frequency",
            derivative=lambda x, p: (
                np.pi * x * np.sin(2 * np.pi * p["frequency"] * x)
            )[:, None],
        )
    nuisance = CycleNuisances(
        np.array(
            [
                [[0.8, 0.15, 0.05], [0.2, 0.65, 0.15]],
                [[0.65, 0.25, 0.1], [0.05, 0.9, 0.05]],
                [[0.05, 0.05, 0.9], [0.15, 0.15, 0.7]],
            ]
        ),
        np.array([[-1.0, 0.0], [1.0, 0.2], [0.0, 1.8]]),
        0.25,
        np.array([0.665, 0.285, 0.05]),
        np.array([0.95, 0.8, 0.95]),
        np.array([0.63, 0.27, 0.1]),
    )
    return model, nuisance


def exact_sums(
    model: CycleModel, nuisance: CycleNuisances, points: int = 9
) -> CycleSums:
    axis = np.linspace(0.05, 3.0, points)
    moments = curve_moments(model, axis, {model.parameter_names[0]: 1.2}, nuisance)
    counts = np.full(points, 3000.0)
    products = np.einsum(
        "n,nij->ij",
        counts,
        moments.covariance + moments.means[:, :, None] * moments.means[:, None, :],
    )
    return CycleSums(axis, counts, counts[:, None] * moments.means, products)


@pytest.mark.parametrize("kind", ["t1", "rabi"])
def test_exact_moments_recover_curve_and_joint_predictions(kind: str) -> None:
    model, nuisance = planted_model(kind)
    sums = exact_sums(model, nuisance)

    name = model.parameter_names[0]
    fit = fit_cycle(sums, model, {name: 1.1})

    np.testing.assert_allclose(fit.parameters[name], 1.2, rtol=1e-5)
    fitted_nuisance = expect_curve_nuisances(fit)
    predicted = curve_moments(model, sums.axis, fit.parameters, fitted_nuisance)
    np.testing.assert_allclose(
        predicted.means, sums.feature_sums / sums.cycle_count[:, None], atol=1e-6
    )
    assert fit.errors[name] > 0
    if kind == "rabi":
        assert fit.reliability == pytest.approx(
            1 / (2 * np.pi * np.ptp(sums.axis) * fit.errors[name])
        )
    else:
        assert fit.reliability is None
    assert fit.covariance.shape == (len(fit.parameter_names),) * 2
    assert fit.J < 1e-7
    assert fit.sharing.p is None
    assert fit.curve.p is not None
    assert fit.populations.shape == fit.population_errors.shape == (9, 3)
    np.testing.assert_allclose(fit.populations.sum(axis=1), 1)
    # A simultaneous latent-state permutation leaves the model unchanged.
    distances = np.sum(
        (nuisance.centers[:, None] - fit.nuisances.centers[None]) ** 2, axis=2
    )
    order = linear_sum_assignment(distances)[1]
    np.testing.assert_allclose(
        fit.nuisances.centers[order], nuisance.centers, atol=1e-5
    )
    np.testing.assert_allclose(
        fitted_nuisance.drive_endpoints[order][:, :, order],
        nuisance.drive_endpoints,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        fit.nuisances.thermal[order], nuisance.thermal, atol=1e-5
    )
    np.testing.assert_allclose(
        fit.nuisances.read_retention[order], nuisance.read_retention, atol=1e-4
    )
    np.testing.assert_allclose(
        fit.nuisances.read_landing[order], nuisance.read_landing, atol=1e-4
    )
    np.testing.assert_allclose(fit.nuisances.variance, nuisance.variance, atol=1e-5)
    # The free pointwise fit's populations match the curve model's at the truth.
    truth_populations = model.populations(sums.axis, {name: 1.2}, nuisance)
    np.testing.assert_allclose(fit.populations[:, order], truth_populations, atol=1e-5)


def test_per_state_curves_fit_their_parameters_jointly() -> None:
    _, nuisance = planted_model()
    model = CycleModel(
        lambda x, p: np.column_stack(
            [np.exp(-x / p["tau"]), np.exp(-x / p["tau"]), np.exp(-x / p["leak_tau"])]
        ),
        ("tau", "leak_tau"),
        ((0.2, 5.0), (0.2, 5.0)),
        derivative=lambda x, p: np.stack(
            [
                np.column_stack(
                    [x * np.exp(-x / p["tau"]) / p["tau"] ** 2] * 2 + [np.zeros_like(x)]
                ),
                np.column_stack(
                    [np.zeros_like(x)] * 2
                    + [x * np.exp(-x / p["leak_tau"]) / p["leak_tau"] ** 2]
                ),
            ],
            axis=2,
        ),
    )
    axis = np.linspace(0.05, 3.0, 11)
    moments = curve_moments(model, axis, {"tau": 1.2, "leak_tau": 0.65}, nuisance)
    sums = CycleSums(
        axis, np.full(len(axis), 3000.0), 3000 * moments.means, np.zeros((65, 65))
    )

    result = fit_cycle(sums, model, {"tau": 1.1, "leak_tau": 0.7})

    np.testing.assert_allclose(
        [result.parameters["tau"], result.parameters["leak_tau"]],
        [1.2, 0.65],
        atol=1e-5,
    )
    assert result.errors["tau"] > 0
    assert result.errors["leak_tau"] > 0


def test_confounded_frequency_has_nan_error_and_no_reliability() -> None:
    rabi, nuisance = planted_model("rabi")
    model = CycleModel(
        lambda x, p: rabi.curve(x, {"frequency": p["frequency"] + p["offset"]}),
        ("frequency", "offset"),
        ((0.1, 2.0), (-0.5, 0.5)),
        oscillation_frequency="frequency",
        derivative=lambda x, p: np.repeat(
            rabi.derivative(x, {"frequency": p["frequency"] + p["offset"]}),
            2,
            axis=2,
        ),
    )
    sums = exact_sums(rabi, nuisance)

    result = fit_cycle(sums, model, {"frequency": 1.2, "offset": 0.0})

    # Only the sum is identified; neither coordinate has a standard error.
    assert np.isnan(result.errors["frequency"])
    assert np.isnan(result.errors["offset"])
    assert result.reliability is None
    np.testing.assert_allclose(
        result.parameters["frequency"] + result.parameters["offset"], 1.2, rtol=1e-5
    )
    # Curve-coordinate ambiguity does not remove pointwise population information.
    assert np.isfinite(result.population_errors).all()
    assert (result.population_errors > 0).all()


def test_endpoint_rank_loss_preserves_frequency_and_population_errors() -> None:
    rabi, nuisance = planted_model("rabi")

    def curve(x, p):
        values = rabi.curve(x, p).copy()
        values[:, 2] = 0.5
        return values

    def derivative(x, p):
        jacobian = rabi.derivative(x, p).copy()
        jacobian[:, 2] = 0.0
        return jacobian

    # Only the midpoint of the third row's endpoints is constrained.
    model = replace(rabi, curve=curve, derivative=derivative)
    sums = exact_sums(model, nuisance)

    result = fit_cycle(sums, model, {"frequency": 1.2})

    np.testing.assert_allclose(result.parameters["frequency"], 1.2, rtol=1e-5)
    assert 0 < result.errors["frequency"] < np.inf
    assert result.reliability == pytest.approx(
        1 / (2 * np.pi * np.ptp(sums.axis) * result.errors["frequency"])
    )
    assert np.isfinite(result.population_errors).all()
    assert (result.population_errors > 0).all()
    distances = np.sum(
        (nuisance.centers[:, None] - result.nuisances.centers[None]) ** 2, axis=2
    )
    order = linear_sum_assignment(distances)[1]
    np.testing.assert_allclose(
        result.populations[:, order],
        model.populations(sums.axis, {"frequency": 1.2}, nuisance),
        atol=1e-5,
    )


def test_wald_weights_do_not_use_pooled_measured_products() -> None:
    model, nuisance = planted_model()
    sums = exact_sums(model, nuisance)
    original = fit_cycle(sums, model, {"tau": 1.1})
    changed = fit_cycle(
        replace(sums, product_sum=np.zeros((65, 65))), model, {"tau": 1.1}
    )
    np.testing.assert_allclose(original.covariance, changed.covariance)
    np.testing.assert_allclose(original.population_errors, changed.population_errors)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["t1", "rabi"])
@pytest.mark.parametrize("contrast,count", [(1.76, 500), (0.176, 30000)])
@pytest.mark.parametrize("seed", [761, 762, 763, 764])
def test_fixed_seed_shot_wald_intervals_in_identifiable_regimes(
    kind: str, contrast: float, count: int, seed: int
) -> None:
    model, nuisance = planted_model(kind)
    centers = np.array(
        [
            [0.0, 0.0],
            [contrast, 0.0],
            1.4 * contrast * np.array([np.cos(1.6), np.sin(1.6)]),
        ]
    )
    nuisance = replace(nuisance, centers=centers, variance=1.0)
    name = model.parameter_names[0]
    sums = sampled_sums(
        model, nuisance, np.linspace(0.05, 3.0, 13), {name: 1.2}, count, seed
    )

    result = fit_cycle(sums, model, {name: 1.1})

    # Four fixed replicates per regime check local Wald scale, not precise calibration.
    assert abs(result.parameters[name] - 1.2) < 1.96 * result.errors[name]
    assert result.within_point.J >= 0
    assert result.sharing.p is None
    assert np.isfinite(result.population_errors).all()


@pytest.mark.parametrize("init", [{}, {"tau": 0.01}, {"tau": np.nan}])
def test_fit_rejects_missing_or_out_of_bounds_starts(init: dict[str, float]) -> None:
    model, nuisance = planted_model()
    sums = exact_sums(model, nuisance, points=3)
    with pytest.raises(ValueError, match="parameter|bounds"):
        fit_cycle(sums, model, init)


@pytest.mark.parametrize("values", [np.full((3, 3), 1.1), np.zeros((3, 2))])
def test_fit_rejects_bad_author_curve(values: np.ndarray) -> None:
    model, nuisance = planted_model()
    sums = exact_sums(model, nuisance, points=3)
    malformed = replace(model, curve=lambda x, p: values)
    with pytest.raises(ValueError, match="curve"):
        fit_cycle(sums, malformed, {"tau": 1.2})
