"""Dominant-pair information and its free-coordinate derivative."""

import numpy as np
from numpy.typing import NDArray

from ._estimator import physical_coordinate_jacobian, physical_coordinates
from ._model import FloatArray, ModelParameters


def pair_information_gradient(
    coordinates: FloatArray, pair: NDArray[np.int64]
) -> FloatArray:
    """Analytic (12,) gradient of pair_information in bounded free coordinates.

    Hold the two state indices in pair fixed. Coordinates follow the estimator's
    free_coordinates order. Translation and rotation leave information unchanged:
    only q = distance²/(4 variance), mean retention L and half-difference m enter.
    Along the pair axis the drive response is -2 q L and the cross-product
    variance is D = (q+1)(q(1-m²)+1)-q²L². Thus information is 4q²L²/D;
    perpendicular Gaussian components have zero response and no cross covariance.
    """
    model = physical_coordinates(coordinates)
    tangent = physical_coordinate_jacobian(coordinates)
    first, second = pair
    delta = model.centres[second] - model.centres[first]
    q = float(delta @ delta / (4 * model.variance))
    mean = float(model.retention[pair].mean())
    difference = float((model.retention[second] - model.retention[first]) / 2)
    numerator = 4 * q**2 * mean**2
    denominator = (q + 1) * (q * (1 - difference**2) + 1) - q**2 * mean**2
    dq = (tangent.centres[:, second] - tangent.centres[:, first]) @ delta / (
        2 * model.variance
    ) - q * tangent.variance / model.variance
    dmean = tangent.retention[:, pair].mean(axis=1)
    ddifference = (tangent.retention[:, second] - tangent.retention[:, first]) / 2
    dnumerator = 8 * q * mean**2 * dq + 8 * q**2 * mean * dmean
    ddenominator = (
        (2 * q * (1 - difference**2 - mean**2) + 2 - difference**2) * dq
        - 2 * q**2 * mean * dmean
        - 2 * q * (q + 1) * difference * ddifference
    )
    return (dnumerator - numerator / denominator * ddenominator) / denominator


def pair_information(model: ModelParameters, pair: NDArray[np.int64]) -> float:
    """Two-state equal-population, symmetrised-landing RDR information.

    pair holds the indices of the two states. The four centred IQ cross
    moments form the observable; Gaussian read noise has the fitted variance
    per component. The two retentions λ may differ.
    """
    centres = model.centres[pair]
    variance = model.variance
    retention = model.retention[pair]
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
