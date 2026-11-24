"""Saved-summary joint fitting of three-state cycles."""

from dataclasses import replace

import numpy as np
import pytest
import zcu_tools.analysis.readout_chain as chain
from scipy.optimize import linear_sum_assignment
from zcu_tools.analysis.readout_chain import (
    CycleFit,
    CycleModel,
    CycleNuisances,
    CycleSums,
    fit_cycle,
    rabi_model,
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
    assert fit.df == 9 * 65 - 27
    assert fit.j_per_df == pytest.approx(fit.J / 558)
    assert fit.expected_band == pytest.approx((0.88026313, 1.11973687))
    assert fit.tail_fraction == 0.0
    np.testing.assert_allclose(fit.predicted_means, sums.feature_sums / 3000, atol=1e-6)
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


def test_rabi_recovers_chain_period_despite_near_nyquist_first_read() -> None:
    model = rabi_model()
    _, nuisance = planted_model("rabi")
    axis = np.linspace(0.0, 3.0, 17)
    moments = curve_moments(model, axis, {"f": 1.2}, nuisance)
    counts = np.full(axis.shape, 250.0)
    means = moments.means.copy()
    # A pickup on the first read dominates its period estimate, but the rest
    # of the features still carry the planted Rabi period.
    first_read_frequency = 2.55
    means[:, 0] += 0.65 * np.cos(2 * np.pi * first_read_frequency * axis)
    products = counts @ moments.covariance.reshape(len(axis), -1)
    sums = CycleSums(
        axis,
        counts,
        counts[:, None] * means,
        products.reshape(65, 65) + means.T @ (counts[:, None] * means),
    )

    fit = fit_cycle(sums, model, {"f": first_read_frequency})

    assert 0.5 / fit.parameters["f"] == pytest.approx(0.4167, abs=0.01)


def test_joint_predictions_use_the_signed_live_projection_on_points_and_grid() -> None:
    model, nuisance = planted_model("rabi")
    axis = np.array([1.1, 0.05, 3.0, 0.7, 1.9])
    counts = 1000 + 500 * axis
    measured = curve_moments(model, axis, {"frequency": 1.2}, nuisance).means.copy()
    measured[0, 2] += 18 / np.sqrt(counts[0])
    # Give the recorded features known unequal pooled noise scales.
    sigma = np.linspace(0.4, 2.0, 65)
    sums = CycleSums(
        axis,
        counts,
        counts[:, None] * measured,
        measured.T @ (counts[:, None] * measured)
        + (counts.sum() - len(counts)) * np.diag(sigma**2),
    )
    fit = fit_cycle(sums, model, {"frequency": 1.1})
    grid = np.linspace(axis.min(), axis.max(), 201)

    projection = chain.cycle_residual_projection(sums, model, fit, model_axis=grid)

    live = chain.cycle_curves(sums).main_sigma
    np.testing.assert_allclose(projection.measured_sigma, live, atol=1e-9)

    fitted_nuisance = expect_curve_nuisances(fit)
    moments = curve_moments(model, axis, fit.parameters, fitted_nuisance)
    # On this almost noise-free fixture, the covariance at the returned fit
    # approximates the frozen last-stage covariance, not the pooled live metric.
    delta = (measured - moments.means) * np.sqrt(counts[:, None])
    residual = np.linalg.solve(np.linalg.cholesky(moments.covariance), delta[..., None])
    assert pytest.approx(np.sum(residual**2), rel=0.01) == fit.J
    assert fit.tail_fraction == pytest.approx(
        np.count_nonzero(np.abs(residual) > 3) / 325
    )
    assert fit.tail_fraction > 0
    assert fit.j_per_df > fit.expected_band[1]
    # Recover the known-covariance PCA from the feature scatter eigenvectors.
    observed = measured / sigma * np.sqrt(counts[:, None])
    background = np.median(measured, axis=0)
    observed -= background / sigma * np.sqrt(counts[:, None])
    _, eigenvectors = np.linalg.eigh(np.cov(observed, rowvar=False))
    loading = eigenvectors[:, -1]
    loading *= np.sign(live @ (observed @ loading))
    difference = (measured - fit.predicted_means) @ (loading / sigma)
    np.testing.assert_allclose(
        projection.measured_sigma - projection.predicted_sigma,
        difference * np.sqrt(counts),
        atol=1e-8,
    )
    sample = grid[[0, 100, -1]]
    grid_means = curve_moments(model, sample, fit.parameters, fitted_nuisance).means
    assert projection.model_sigma.shape == grid.shape
    np.testing.assert_allclose(
        projection.model_sigma[[0, 100, -1]],
        ((grid_means - background) @ (loading / sigma)) * np.sqrt(1000 + 500 * sample),
        atol=1e-8,
    )


@pytest.fixture
def saved_projection_case() -> tuple[CycleSums, CycleModel, CycleFit]:
    model, nuisance = planted_model("rabi")
    axis = np.arange(3.0)
    counts = np.full(3, 16.0)
    feature_sums = np.zeros((3, 65))
    feature_sums[:, 2] = counts
    # Exact rank one after removing the constant third feature's mean square.
    products = np.zeros((65, 65))
    products[:2, :2] = 45
    products[2, 2] = 48
    sums = CycleSums(axis, counts, feature_sums, products)
    fit = CycleFit(
        parameters={"frequency": 1.2},
        errors={},
        nuisances=nuisance,
        covariance=np.empty((0, 0)),
        parameter_names=(),
        reliability=None,
        J=0.0,
        df=168,
        j_per_df=0.0,
        expected_band=(0.78, 1.22),
        tail_fraction=0.0,
        predicted_means=model.feature_means(axis, {"frequency": 1.2}, nuisance),
        relaxation=None,
    )
    return sums, model, fit


@pytest.mark.parametrize("moments", ["rank-one", "rank-zero", "one-ulp"])
def test_projection_names_compatible_rank_loss_without_changing_live(
    saved_projection_case: tuple[CycleSums, CycleModel, CycleFit], moments: str
) -> None:
    sums, model, fit = saved_projection_case
    if moments == "rank-zero":
        sums.product_sum[:2, :2] = 0
    elif moments == "one-ulp":
        sums.product_sum[2, 2] = np.nextafter(48.0, -np.inf)

    with pytest.raises(ValueError, match="positive definite") as live_error:
        chain.cycle_curves(sums)
    assert type(live_error.value) is ValueError
    with pytest.raises(chain.CycleProjectionUnavailable):
        chain.cycle_residual_projection(
            sums, model, fit, model_axis=np.linspace(0, 2, 5)
        )


@pytest.mark.parametrize(
    "moments",
    [
        "negative",
        "low-scale",
        "subnormal",
        "antisymmetric",
        "infinite",
        "zero-scale-cross",
    ],
)
def test_projection_does_not_classify_bad_saved_moments_as_rank_loss(
    saved_projection_case: tuple[CycleSums, CycleModel, CycleFit], moments: str
) -> None:
    sums, model, fit = saved_projection_case
    if moments == "negative":
        sums.product_sum[2, 2] -= 1e-9
    elif moments == "low-scale":
        sums.product_sum[64, 64] = -1e-150
    elif moments == "subnormal":
        sums.product_sum[64, 64] = np.nextafter(0.0, -np.inf)
    elif moments == "antisymmetric":
        sums.product_sum[0, 1] += 1e-9
    elif moments == "infinite":
        sums.product_sum[0, 0] = np.inf
    else:
        sums.product_sum[0, 64] = sums.product_sum[64, 0] = 1e-12

    with pytest.raises(ValueError, match="positive definite|finite") as error:
        chain.cycle_residual_projection(sums, model, fit)
    assert type(error.value) is ValueError


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("prediction-shape", "acquired feature shape"),
        ("prediction-nan", "predictions.*finite"),
        ("grid", "axis.*finite"),
        ("model", "nuisance type"),
        ("grid-prediction", "curve.*shape"),
        ("callback", "callback error"),
    ],
)
def test_projection_checks_independent_errors_even_when_whitening_cannot_succeed(
    saved_projection_case: tuple[CycleSums, CycleModel, CycleFit],
    invalid: str,
    message: str,
) -> None:
    sums, curve_model, fit = saved_projection_case
    model: CycleModel | chain.RateModel = curve_model
    grid = np.linspace(0, 2, 5)
    callback_error = ValueError("callback error")

    def fail_callback(*_args: object) -> np.ndarray:
        raise callback_error

    if invalid == "prediction-shape":
        fit = replace(fit, predicted_means=np.zeros((3, 64)))
    elif invalid == "prediction-nan":
        fit.predicted_means[0, 0] = np.nan
    elif invalid == "grid":
        grid[0] = np.inf
    elif invalid == "model":
        model = chain.t1_model()
    elif invalid == "grid-prediction":
        model = replace(curve_model, curve=lambda *_args: np.zeros((5, 2)))
    else:
        model = replace(curve_model, curve=fail_callback)

    with pytest.raises(ValueError, match=message) as error:
        chain.cycle_residual_projection(sums, model, fit, model_axis=grid)
    assert type(error.value) is ValueError
    if invalid == "callback":
        assert error.value is callback_error


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


def test_endpoint_rank_loss_preserves_frequency_error() -> None:
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


def test_wald_weights_do_not_use_pooled_measured_products() -> None:
    model, nuisance = planted_model()
    sums = exact_sums(model, nuisance)
    original = fit_cycle(sums, model, {"tau": 1.1})
    changed = fit_cycle(
        replace(sums, product_sum=np.zeros((65, 65))), model, {"tau": 1.1}
    )
    np.testing.assert_allclose(original.covariance, changed.covariance)
    assert original.errors == changed.errors
    assert original.J == changed.J
    assert original.tail_fraction == changed.tail_fraction


def test_low_contrast_fit_returns_joint_estimate_and_residual_numbers() -> None:
    model, nuisance = planted_model()
    nuisance = replace(
        nuisance,
        variance=1.0,
        centers=np.array([[0.0, 0.0], [0.35, 0.0], [-0.014, 0.49]]),
    )
    sums = sampled_sums(
        model, nuisance, np.linspace(0.05, 3.0, 5), {"tau": 1.2}, 500, 761
    )

    result = fit_cycle(sums, model, {"tau": 1.1})

    assert 0.2 <= result.parameters["tau"] <= 5.0
    assert 0 < result.J < np.inf
    assert result.df == 298
    assert result.j_per_df == pytest.approx(result.J / 298)
    assert 0 <= result.tail_fraction <= 1
    assert np.isfinite(result.predicted_means).all()


def test_fit_raises_when_the_joint_model_covariance_is_numerically_singular() -> None:
    model, nuisance = planted_model()
    # At this ADC offset the shot variance of the second moments cannot be
    # resolved by subtracting their raw quadratic moments in float64.
    nuisance = replace(nuisance, centers=nuisance.centers + 1e8)
    sums = exact_sums(model, nuisance, points=3)

    with pytest.raises(np.linalg.LinAlgError):
        fit_cycle(sums, model, {"tau": 1.1})


def shot_fit(kind: str, contrast: float, count: int, seed: int) -> tuple[str, CycleFit]:
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

    return name, fit_cycle(sums, model, {name: 1.1})


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["t1", "rabi"])
@pytest.mark.parametrize("contrast,count", [(1.76, 500), (0.176, 30000)])
@pytest.mark.parametrize("seed", [761, 762, 763, 764])
def test_fixed_seed_shot_wald_intervals_in_identifiable_regimes(
    kind: str, contrast: float, count: int, seed: int
) -> None:
    name, result = shot_fit(kind, contrast, count, seed)

    # Four fixed replicates per regime check local Wald scale, not precise calibration.
    assert abs(result.parameters[name] - 1.2) < 1.96 * result.errors[name]


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
