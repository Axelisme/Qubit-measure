"""Closed-form three-state start, coordinate maps and bounded refinement."""

from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

from ._model import (
    EARLY_TRIPLES,
    LAG_PAIRS,
    LATE_TRIPLES,
    MEANS,
    SAME_READ,
    FloatArray,
    ModelParameters,
    ModelTangents,
    contiguous_covariance,
    moment_mean_jacobian,
    moment_means,
    stationary_population,
)

# The 25 scalar moments along one IQ axis (projected_moments), in order.
AXIS_MEAN = 0
AXIS_SAME_READ = 1
AXIS_LAGS = slice(2, 17)  # lag 1..15 x_t x_{t+k}
AXIS_EARLY = slice(17, 21)  # lag 1..4 x_t² x_{t+k}
AXIS_LATE = slice(21, 25)  # lag 1..4 x_t x_{t+k}²

# The 12 free coordinates of the bounded fit, in order.
FREE_CENTRES = slice(0, 6)  # centres of g, e, f, each I then Q
FREE_THIRD_CENTRE = slice(4, 6)
FREE_LOG_VARIANCE = 6
FREE_LOGIT_RETENTION = slice(7, 10)
FREE_THIRD_LOGIT_RETENTION = 9
FREE_LOG_LANDING_RATIOS = slice(10, 12)  # log(ρ_e/ρ_g), log(ρ_f/ρ_g)
FREE_THIRD_LOG_LANDING_RATIO = 11
FREE_COUNT = 12


class UnfittableMoments(ValueError):
    """The saved moments of one setting cannot be fitted by the estimator."""


def second_moments(features: FloatArray) -> FloatArray:
    """Same-read matrix followed by the fifteen lagged second-moment matrices."""
    ii, iq, qq = features[SAME_READ]
    same = np.array([[ii, iq], [iq, qq]])
    return np.concatenate((same[None], features[LAG_PAIRS].reshape(15, 2, 2)))


def projected_moments(features: FloatArray, direction: FloatArray) -> FloatArray:
    """The 25 scalar moments along an IQ unit vector, without losing cross terms."""
    quadratic = np.array(
        [direction[0] ** 2, 2 * direction[0] * direction[1], direction[1] ** 2]
    )
    second = np.einsum("i,kij,j->k", direction, second_moments(features), direction)
    early = np.einsum(
        "i,kij,j->k", quadratic, features[EARLY_TRIPLES].reshape(4, 3, 2), direction
    )
    late = np.einsum(
        "i,kij,j->k", direction, features[LATE_TRIPLES].reshape(4, 2, 3), quadratic
    )
    return np.concatenate(([direction @ features[MEANS]], second, early, late))


def projected_cumulants(
    axis_moments: FloatArray, retention: float
) -> tuple[float, float]:
    """Second and third scalar cumulants from lag-power projections."""
    mean = axis_moments[AXIS_MEAN]
    same_read = axis_moments[AXIS_SAME_READ]
    lags = axis_moments[AXIS_LAGS]
    covariance = lags - mean**2
    powers = retention ** np.arange(1, 16)
    second = float(covariance @ powers / (powers @ powers))
    early = (
        axis_moments[AXIS_EARLY] - 2 * mean * lags[:4] - mean * same_read + 2 * mean**3
    )
    late = (
        axis_moments[AXIS_LATE] - 2 * mean * lags[:4] - mean * same_read + 2 * mean**3
    )
    powers3 = retention ** np.arange(1, 5)
    third = float((early + late) @ powers3 / (2 * (powers3 @ powers3)))
    return second, third


def lag_retention(
    axis_moments: FloatArray, lower: float, upper: float, unobserved: float
) -> float:
    """Closed-form lag-shift least-squares retention seed, clipped to start bounds.

    unobserved is the seed when lags 1..14 carry no covariance at all.
    """
    covariance = axis_moments[AXIS_LAGS] - axis_moments[AXIS_MEAN] ** 2
    denominator = float(covariance[:-1] @ covariance[:-1])
    if denominator == 0:
        return unobserved
    # A positive sum below 1e-300 divides by 1e-300 instead.
    ratio = covariance[:-1] @ covariance[1:] / max(denominator, 1e-300)
    return float(np.clip(ratio, lower, upper))


def main_axis_start(
    axis_moments: FloatArray,
) -> tuple[float, float, float, float, float]:
    """Lower centre, distance, variance, retention and upper population on one axis."""
    retention = lag_retention(axis_moments, 0.02, 0.98, unobserved=0.02)
    second, third = projected_cumulants(axis_moments, retention)
    second = max(second, 1e-9)
    fraction_product = 1 / (third**2 / second**3 + 4)
    distance = float(np.sqrt(second / fraction_product))
    population = float(
        np.clip(
            0.5 * (1 - np.sign(third) * np.sqrt(max(1 - 4 * fraction_product, 0))),
            0.01,
            0.99,
        )
    )
    mean = axis_moments[AXIS_MEAN]
    variance = float(max(axis_moments[AXIS_SAME_READ] - mean**2 - second, 1e-3))
    return (
        float(mean - population * distance),
        distance,
        variance,
        retention,
        population,
    )


def closed_form_start(features: FloatArray) -> ModelParameters:
    """Two states on the principal signal axis and a third on the orthogonal axis.

    The signal matrix sums the lag 1..3 covariances. Its leading eigenvector
    carries the two-state solution of the projected second and third
    cumulants; the orthogonal axis supplies the third state's height,
    population and retention. Raise UnfittableMoments without a positive
    signal eigenvalue.
    """
    mean = features[MEANS]
    second = second_moments(features)
    signal = np.sum(second[1:4] - np.outer(mean, mean), axis=0)
    eigenvalues, axes = np.linalg.eigh((signal + signal.T) / 2)
    if eigenvalues[-1] <= 0:
        raise UnfittableMoments("readout moments have no identifiable main-axis signal")
    main, orthogonal = axes[:, 1], axes[:, 0]
    centre, distance, variance, retention, population = main_axis_start(
        projected_moments(features, main)
    )
    orthogonal_moments = projected_moments(features, orthogonal)
    # An orthogonal mode without lag covariance starts at the prior's λ 0.9.
    retention_f = lag_retention(orthogonal_moments, 0.05, 0.99, unobserved=0.9)
    amplitude, skew = projected_cumulants(orthogonal_moments, retention_f)
    if amplitude > 1e-9 and abs(skew) > 1e-12:
        fraction_product = 1 / (skew**2 / amplitude**3 + 4)
        population_f = 0.5 * (1 - np.sqrt(max(1 - 4 * fraction_product, 0)))
        height = np.sign(skew) * np.sqrt(amplitude / fraction_product)
    else:
        population_f, height = 1e-3, np.sqrt(max(amplitude, 1e-6) / 1e-3)
    population_f = float(np.clip(population_f, 1e-3, 0.3))
    cross = main @ (second[0] - np.outer(mean, mean)) @ orthogonal
    along_f = main @ mean + cross / (population_f * height)
    baseline = orthogonal_moments[AXIS_MEAN] - population_f * height
    centres = np.array(
        [
            centre * main + baseline * orthogonal,
            (centre + distance) * main + baseline * orthogonal,
            along_f * main + (baseline + height) * orthogonal,
        ]
    )
    stationary = np.array(
        [
            (1 - population) * (1 - population_f),
            population * (1 - population_f),
            population_f,
        ]
    )
    retentions = np.array([retention, retention, retention_f])
    landing = stationary * (1 - retentions)
    return ModelParameters(centres, variance, retentions, landing / landing.sum())


def prior_coordinates(model: ModelParameters) -> tuple[FloatArray, FloatArray]:
    """Fixed weak priors in free coordinates: location and scale, inf = no prior.

    The third centre is pulled toward the g-e midpoint plus one g-e distance
    perpendicular (on the side of the current third centre), with that
    distance as scale; logit λ_f toward logit 0.9 (scale 1.5); log(ρ_f/ρ_g)
    toward log 0.005 (scale 2).
    """
    g, e, f = model.centres
    distance = float(np.linalg.norm(e - g))
    direction = (e - g) / distance
    perpendicular = np.array([-direction[1], direction[0]])
    midpoint = (g + e) / 2
    if (f - midpoint) @ perpendicular < 0:
        perpendicular = -perpendicular
    location = np.zeros(FREE_COUNT)
    scale = np.full(FREE_COUNT, np.inf)
    location[FREE_THIRD_CENTRE] = midpoint + distance * perpendicular
    scale[FREE_THIRD_CENTRE] = distance
    location[FREE_THIRD_LOGIT_RETENTION] = np.log(9)
    scale[FREE_THIRD_LOGIT_RETENTION] = 1.5
    location[FREE_THIRD_LOG_LANDING_RATIO] = np.log(0.005)
    scale[FREE_THIRD_LOG_LANDING_RATIO] = 2
    return location, scale


def safe_start(features: FloatArray) -> ModelParameters:
    """Move the third state to its prior when the closed-form start cannot see it.

    That happens when the start is non-finite, the third landing is at most
    1.5e-3, or the third centre lies over 50 g-e distances from the origin.
    The third state then gets the prior centre, λ 0.9 and landing 0.005.
    """
    model = closed_form_start(features)
    distance = np.linalg.norm(model.centres[1] - model.centres[0])
    if (
        not model.is_finite()
        or model.landing[2] <= 1.5e-3
        or np.linalg.norm(model.centres[2]) > 50 * distance
    ):
        prior, _ = prior_coordinates(model)
        centres = model.centres.copy()
        centres[2] = prior[FREE_THIRD_CENTRE]
        retention = model.retention.copy()
        retention[2] = 0.9
        landing = model.landing.copy()
        landing[2] = 0.005
        model = ModelParameters(
            centres, model.variance, retention, landing / landing.sum()
        )
    return model


def free_coordinates(model: ModelParameters) -> FloatArray:
    """Centres, log variance, logit retentions and two log landing ratios."""
    centres, variance, retention, landing = model
    return np.concatenate(
        (
            centres.ravel(),
            [np.log(variance)],
            np.log(retention / (1 - retention)),
            np.log(landing[1:] / landing[0]),
        )
    )


def physical_coordinates(coordinates: FloatArray) -> ModelParameters:
    """Inverse of free_coordinates, bounded: λ in [0.01, 0.995], landing >= 1e-5."""
    retention = np.clip(expit(coordinates[FREE_LOGIT_RETENTION]), 0.01, 0.995)
    ratios = coordinates[FREE_LOG_LANDING_RATIOS]
    weights = np.exp(np.clip(np.array([0, ratios[0], ratios[1]]), -12, 12))
    landing = np.maximum(weights / weights.sum(), 1e-5)
    return ModelParameters(
        centres=coordinates[FREE_CENTRES].reshape(3, 2).copy(),
        variance=float(np.exp(coordinates[FREE_LOG_VARIANCE])),
        retention=retention,
        landing=landing / landing.sum(),
    )


def clip_slope(values: FloatArray, lower: float, upper: float) -> FloatArray:
    """Slope of clipping, with the mean of one-sided slopes at either bound."""
    # At a kink use the mean slope to preserve the estimator's symmetric convention.
    # Log/exp round trips can place an exact bound a few floating-point ulps away.
    tolerance = 8 * np.finfo(float).eps
    at_bound = np.isclose(values, lower, rtol=tolerance, atol=0) | np.isclose(
        values, upper, rtol=tolerance, atol=0
    )
    return np.where(at_bound, 0.5, ((values > lower) & (values < upper)).astype(float))


def physical_coordinate_jacobian(coordinates: FloatArray) -> ModelTangents:
    """Physical tangents along the 12 free coordinates, including bounds.

    Coordinates have the order documented by free_coordinates. Clip and floor
    derivatives vanish outside their bounds and use mean one-sided slopes at
    exact kinks. Landing tangents include both normalisations around the floor.
    """
    basis = np.eye(FREE_COUNT)
    raw_retention = expit(coordinates[FREE_LOGIT_RETENTION])
    dretention = (
        basis[:, FREE_LOGIT_RETENTION]
        * raw_retention
        * (1 - raw_retention)
        * clip_slope(raw_retention, 0.01, 0.995)
    )
    ratios = np.array([0.0, *coordinates[FREE_LOG_LANDING_RATIOS]])
    weights = np.exp(np.clip(ratios, -12, 12))
    dweights = np.column_stack(
        (np.zeros(FREE_COUNT), basis[:, FREE_LOG_LANDING_RATIOS])
    )
    dweights *= weights * clip_slope(ratios, -12, 12)
    landing = weights / weights.sum()
    dlanding = (dweights - landing * dweights.sum(axis=1)[:, None]) / weights.sum()
    dlanding *= clip_slope(landing, 1e-5, np.inf)
    floored = np.maximum(landing, 1e-5)
    normalised = floored / floored.sum()
    dlanding = (dlanding - normalised * dlanding.sum(axis=1)[:, None]) / floored.sum()
    return ModelTangents(
        centres=basis[:, FREE_CENTRES].reshape(FREE_COUNT, 3, 2),
        variance=np.exp(coordinates[FREE_LOG_VARIANCE]) * basis[:, FREE_LOG_VARIANCE],
        retention=dretention,
        landing=dlanding,
    )


def moment_jacobian(coordinates: FloatArray) -> FloatArray:
    """Analytic (113, 12) moment Jacobian in the estimator's bounded free space.

    Coordinates follow free_coordinates. Propagate log variance, clipped logit
    retention and floored log landing-ratio tangents through the exact model.
    """
    return moment_mean_jacobian(
        physical_coordinates(coordinates), physical_coordinate_jacobian(coordinates)
    )


class _ResidualRows:
    """Fixed start-model whitening and prior rows for bounded least squares."""

    def __init__(self, features: FloatArray, blocks: float) -> None:
        self.features = features
        start = safe_start(features)
        self.start = free_coordinates(start)
        landing = start.landing.copy()
        landing[np.argmin(stationary_population(start))] = 0
        weight_model = start._replace(landing=landing / landing.sum())
        covariance = contiguous_covariance(weight_model) / blocks
        eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        self.whitener = (
            vectors / np.sqrt(np.maximum(eigenvalues, eigenvalues.max() * 1e-14))
        ).T
        self.prior, scale = prior_coordinates(start)
        self.prior_indices = np.flatnonzero(np.isfinite(scale))
        self.prior_root = 1 / scale[self.prior_indices]
        self.prior_rows = np.zeros((len(self.prior_indices), FREE_COUNT))
        self.prior_rows[np.arange(len(self.prior_indices)), self.prior_indices] = (
            self.prior_root
        )

    def residual(self, coordinates: FloatArray) -> FloatArray:
        moments = self.whitener @ (
            moment_means(physical_coordinates(coordinates)) - self.features
        )
        prior = (coordinates - self.prior)[self.prior_indices] * self.prior_root
        return np.concatenate((moments, prior))

    def jacobian(self, coordinates: FloatArray) -> FloatArray:
        return np.vstack(
            (self.whitener @ moment_jacobian(coordinates), self.prior_rows)
        )

    def sampling_covariance(self, coordinates: FloatArray) -> FloatArray:
        derivative = self.whitener @ moment_jacobian(coordinates)
        normal = derivative.T @ derivative + self.prior_rows.T @ self.prior_rows
        response = np.linalg.solve(normal, derivative.T)
        # The prior is fixed, not a new random observation in each stream.
        return response @ response.T


def bounded_fit(features: FloatArray, blocks: float) -> tuple[FloatArray, FloatArray]:
    """Refine one setting's (113,) moment averages with bounded trf least squares.

    blocks is the number of contiguous 32-read blocks. Start with safe_start;
    hold its weak priors and model covariance fixed. For weights only, set the
    start's least populated state's landing to zero and renormalise landing.
    Do not change the start, priors or free coordinates. Bound λ in [.01, .995]
    and the two log landing ratios in [-10.8, 10.8]; keep the coordinate map's
    1e-5 landing floor. Return (12,) free coordinates and their (12,12)
    delta-method sampling covariance at the solution, including fixed priors.
    Raise UnfittableMoments for a missing signal, non-finite start or exhausted
    evaluation budget (1200 residual calls); LinAlgError for singular covariance.
    """
    rows = _ResidualRows(features, blocks)
    lower, upper = np.full(FREE_COUNT, -np.inf), np.full(FREE_COUNT, np.inf)
    # Stay inside the clipping kinks, whose tangents use mean one-sided slopes.
    lower[FREE_LOGIT_RETENTION] = logit(0.01) + 1e-9
    upper[FREE_LOGIT_RETENTION] = logit(0.995) - 1e-9
    lower[FREE_LOG_LANDING_RATIOS] = -10.8
    upper[FREE_LOG_LANDING_RATIOS] = 10.8
    start = np.clip(rows.start, lower + 1e-10, upper - 1e-10)
    if not np.isfinite(rows.residual(start)).all():
        raise UnfittableMoments("non-finite residual at the start")
    # scipy-stubs incorrectly restricts jac to str; SciPy accepts a callable.
    analytic: Any = rows.jacobian
    solved = least_squares(
        rows.residual,
        start,
        jac=analytic,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=100 * FREE_COUNT,
    )
    if solved.status == 0:
        raise UnfittableMoments(
            "fit hit the evaluation budget (1200 evaluations, status 0)"
        )
    return solved.x, rows.sampling_covariance(solved.x)
