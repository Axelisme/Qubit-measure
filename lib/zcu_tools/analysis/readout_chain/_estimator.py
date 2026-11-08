"""Closed-form start, fixed weak priors, one Gauss-Newton update and pair score.

The start and prior strengths follow the three-state research estimator. All
113 features enter one update; there is no iterative optimisation or K selection.
"""

import numpy as np
from scipy.special import expit

from ._model import (
    FloatArray,
    contiguous_covariance,
    moment_means,
    stationary_population,
)


def second_moments(features: FloatArray) -> FloatArray:
    """Same-read matrix followed by the fifteen lagged second-moment matrices."""
    same = np.array([[features[2], features[3]], [features[3], features[4]]])
    return np.concatenate((same[None], features[5:65].reshape(15, 2, 2)))


def projected_moments(features: FloatArray, direction: FloatArray) -> FloatArray:
    """The 25 scalar moments along an IQ unit vector, without losing cross terms."""
    quadratic = np.array(
        [direction[0] ** 2, 2 * direction[0] * direction[1], direction[1] ** 2]
    )
    second = np.einsum("i,kij,j->k", direction, second_moments(features), direction)
    left = np.einsum(
        "i,kij,j->k", quadratic, features[65:89].reshape(4, 3, 2), direction
    )
    right = np.einsum(
        "i,kij,j->k", direction, features[89:113].reshape(4, 2, 3), quadratic
    )
    return np.concatenate(([direction @ features[:2]], second, left, right))


def projected_cumulants(features: FloatArray, retention: float) -> tuple[float, float]:
    """Second and third scalar cumulants from lag-power projections."""
    mean = features[0]
    covariance = features[2:17] - mean**2
    powers = retention ** np.arange(1, 16)
    second = float(covariance @ powers / (powers @ powers))
    left = features[17:21] - 2 * mean * features[2:6] - mean * features[1] + 2 * mean**3
    right = (
        features[21:25] - 2 * mean * features[2:6] - mean * features[1] + 2 * mean**3
    )
    powers3 = retention ** np.arange(1, 5)
    third = float((left + right) @ powers3 / (2 * (powers3 @ powers3)))
    return second, third


def lag_retention(features: FloatArray, lower: float, upper: float) -> float:
    """Closed-form lag-shift least-squares retention seed, clipped to start bounds."""
    covariance = features[2:17] - features[0] ** 2
    denominator = max(float(covariance[:-1] @ covariance[:-1]), 1e-300)
    return float(np.clip(covariance[:-1] @ covariance[1:] / denominator, lower, upper))


def main_axis_start(features: FloatArray) -> FloatArray:
    """Scalar centre, distance, variance, retention and population from cumulants."""
    retention = lag_retention(features, 0.02, 0.98)
    second, third = projected_cumulants(features, retention)
    second = max(second, 1e-9)
    fraction_product = 1 / (third**2 / second**3 + 4)
    distance = np.sqrt(second / fraction_product)
    population = float(
        np.clip(
            0.5 * (1 - np.sign(third) * np.sqrt(max(1 - 4 * fraction_product, 0))),
            0.01,
            0.99,
        )
    )
    variance = max(features[1] - features[0] ** 2 - second, 1e-3)
    return np.array(
        [features[0] - population * distance, distance, variance, retention, population]
    )


def closed_form_start(features: FloatArray) -> FloatArray:
    """Research start3: principal-axis pair, orthogonal third state, 13 values."""
    mean = features[:2]
    second = second_moments(features)
    signal = np.sum(second[1:4] - np.outer(mean, mean), axis=0)
    eigenvalues, axes = np.linalg.eigh((signal + signal.T) / 2)
    if eigenvalues[-1] <= 0:
        raise ValueError("readout moments have no identifiable main-axis signal")
    main, orthogonal = axes[:, 1], axes[:, 0]
    centre, distance, variance, retention, population = main_axis_start(
        projected_moments(features, main)
    )
    projected = projected_moments(features, orthogonal)
    retention_f = lag_retention(projected, 0.05, 0.99)
    amplitude, skew = projected_cumulants(projected, retention_f)
    if amplitude > 1e-9 and abs(skew) > 1e-12:
        fraction_product = 1 / (skew**2 / amplitude**3 + 4)
        population_f = 0.5 * (1 - np.sqrt(max(1 - 4 * fraction_product, 0)))
        height = np.sign(skew) * np.sqrt(amplitude / fraction_product)
    else:
        population_f, height = 1e-3, np.sqrt(max(amplitude, 1e-6) / 1e-3)
    population_f = float(np.clip(population_f, 1e-3, 0.3))
    cross = main @ (second[0] - np.outer(mean, mean)) @ orthogonal
    along_f = main @ mean + cross / (population_f * height)
    baseline = projected[0] - population_f * height
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
    return np.concatenate(
        (centres.ravel(), [variance], retentions, landing / landing.sum())
    )


def prior_coordinates(theta: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Fixed weak priors on third centre, logit retention and log landing ratio."""
    centres = theta[:6].reshape(3, 2)
    distance = float(np.linalg.norm(centres[1] - centres[0]))
    direction = (centres[1] - centres[0]) / distance
    orthogonal = np.array([-direction[1], direction[0]])
    midpoint = (centres[0] + centres[1]) / 2
    if (centres[2] - midpoint) @ orthogonal < 0:
        orthogonal = -orthogonal
    location = np.zeros(12)
    scale = np.full(12, np.inf)
    location[4:6], scale[4:6] = midpoint + distance * orthogonal, distance
    location[9], scale[9] = np.log(9), 1.5
    location[11], scale[11] = np.log(0.005), 2
    return location, scale


def safe_start(features: FloatArray) -> FloatArray:
    """Use the prior third centre when its orthogonal mode is unobserved."""
    theta = closed_form_start(features)
    distance = np.linalg.norm(theta[2:4] - theta[:2])
    if (
        not np.isfinite(theta).all()
        or theta[12] <= 1.5e-3
        or np.linalg.norm(theta[4:6]) > 50 * distance
    ):
        prior, _ = prior_coordinates(theta)
        theta[4:6], theta[9] = prior[4:6], 0.9
        landing = theta[10:13].copy()
        landing[2] = 0.005
        theta[10:13] = landing / landing.sum()
    return theta


def free_coordinates(theta: FloatArray) -> FloatArray:
    """Centres, log variance, logit retentions and two log landing ratios."""
    retention, landing = theta[7:10], theta[10:13]
    return np.concatenate(
        (
            theta[:6],
            [np.log(theta[6])],
            np.log(retention / (1 - retention)),
            np.log(landing[1:] / landing[0]),
        )
    )


def physical_coordinates(coordinates: FloatArray) -> FloatArray:
    """Research bounded parameter transform; λ in [.01,.995], positive landing."""
    variance = np.exp(coordinates[6])
    retention = np.clip(expit(coordinates[7:10]), 0.01, 0.995)
    weights = np.exp(np.clip(np.array([0, coordinates[10], coordinates[11]]), -12, 12))
    landing = np.maximum(weights / weights.sum(), 1e-5)
    return np.concatenate(
        (coordinates[:6], [variance], retention, landing / landing.sum())
    )


def moment_jacobian(coordinates: FloatArray) -> FloatArray:
    """Central-difference derivative of all 113 moments in the 12 free coordinates."""
    result = np.empty((113, 12))
    for i in range(12):
        step = 1e-6 * max(abs(coordinates[i]), 1e-2)
        up, down = coordinates.copy(), coordinates.copy()
        up[i] += step
        down[i] -= step
        result[:, i] = (
            moment_means(physical_coordinates(up))
            - moment_means(physical_coordinates(down))
        ) / (2 * step)
    return result


def one_step_fit(features: FloatArray, blocks: float) -> tuple[FloatArray, FloatArray]:
    """One penalised model-weighted GN step and its sampling covariance in free space."""
    start = safe_start(features)
    covariance = contiguous_covariance(start) / blocks
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    whitener = vectors / np.sqrt(np.maximum(eigenvalues, eigenvalues.max() * 1e-14))
    coordinates = free_coordinates(start)
    derivative = whitener.T @ moment_jacobian(coordinates)
    residual = whitener.T @ (features - moment_means(start))
    prior, scale = prior_coordinates(start)
    precision = np.diag(1 / scale**2)
    normal = derivative.T @ derivative + precision
    update = np.linalg.solve(
        normal, derivative.T @ residual + precision @ (prior - coordinates)
    )
    fitted = coordinates + update
    # Frequentist sampling covariance, not inverse penalised Hessian: the fixed
    # prior is not a fresh random observation on every acquired stream.
    response = np.linalg.solve(normal, derivative.T)
    sampling_covariance = response @ response.T
    return fitted, sampling_covariance


def pair_information(theta: FloatArray, pair: np.ndarray) -> float:
    """Two-state equal-population, symmetrised-landing RDR information.

    The four centred IQ cross moments form the observable; Gaussian read noise
    has fitted variance per component. Individual retain/reset λ may differ.
    """
    centres = theta[:6].reshape(3, 2)[pair]
    variance = theta[6]
    retention = theta[7:10][pair]
    uniform = np.full(2, 0.5)
    transition = np.diag(retention) + (1 - retention[:, None]) * uniform
    exchange = np.array([[-1.0, 1.0], [1.0, -1.0]])
    mean_first, mean_second = uniform @ centres, (uniform @ transition) @ centres
    population_covariance = np.diag(uniform) - np.outer(uniform, uniform)
    gradient = (
        centres.T @ population_covariance @ transition @ exchange @ centres
    ).ravel()
    cross_mean = np.zeros((2, 2))
    fourth = np.zeros((4, 4))
    for first in range(2):
        for second in range(2):
            probability = uniform[first] * transition[first, second]
            left, right = centres[first] - mean_first, centres[second] - mean_second
            cross_mean += probability * np.outer(left, right)
            left_second = np.outer(left, left) + variance * np.eye(2)
            right_second = np.outer(right, right) + variance * np.eye(2)
            fourth += probability * np.kron(left_second, right_second)
    covariance = fourth - np.outer(cross_mean.ravel(), cross_mean.ravel())
    return float(gradient @ np.linalg.solve(covariance, gradient))


def fitted_pair_result(coordinates: FloatArray, covariance: FloatArray) -> FloatArray:
    """Metric, standard error, separation, two retentions and least population."""
    theta = physical_coordinates(coordinates)
    if not np.isfinite(theta).all() or theta[6] <= 0:
        raise ValueError("non-finite one-step readout fit")
    population = stationary_population(theta)
    order = np.argsort(-population, kind="stable")
    pair = order[:2]
    information = pair_information(theta, pair)
    gradient = np.empty(12)
    for i in range(12):
        step = 1e-6 * max(abs(coordinates[i]), 1e-2)
        up, down = coordinates.copy(), coordinates.copy()
        up[i] += step
        down[i] -= step
        gradient[i] = (
            pair_information(physical_coordinates(up), pair)
            - pair_information(physical_coordinates(down), pair)
        ) / (2 * step)
    error = np.sqrt(max(float(gradient @ covariance @ gradient), 0))
    centres = theta[:6].reshape(3, 2)
    separation = np.linalg.norm(centres[pair[1]] - centres[pair[0]]) / np.sqrt(theta[6])
    return np.array(
        [
            information,
            error,
            separation,
            theta[7:10][pair[0]],
            theta[7:10][pair[1]],
            population[order[2]],
        ]
    )
