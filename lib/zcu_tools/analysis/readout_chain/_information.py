"""All-lag symmetric-chain information and its analytic derivatives."""

import numpy as np
from numpy.typing import NDArray

from ._estimator import physical_coordinate_jacobian, physical_coordinates
from ._model import FloatArray, ModelParameters


def all_lag_information(q: float, qnd: float) -> float:
    """Information per read about a weak drive in an infinite symmetric DR chain.

    q >= 0 is d²/(4σ²); 0 < qnd < 1 is the symmetric-chain QND.
    Combine products of reads at every lag j >= 1 with GLS weights using their
    exact long-run covariance. Return dimensionless information, zero at q=0.
    """
    return _value_gradient(q, qnd)[0]


def all_lag_information_gradient(q: float, qnd: float) -> FloatArray:
    """Analytic (2,) gradient of all_lag_information in (q, QND) order.

    Inputs have the same domain as all_lag_information. The gradient is zero at
    q=0. No finite differences or lag truncation are used.
    """
    return _value_gradient(q, qnd)[1]


def _value_gradient(q: float, correlation: float) -> tuple[float, FloatArray]:
    if q == 0:
        return 0.0, np.zeros(2)
    # Extended intermediates preserve the small denominator when QND is near one.
    signal, qnd = np.longdouble(q), np.longdouble(correlation)
    kappa = (1 + qnd**2 + signal * (1 - qnd**2)) / (2 * qnd)
    root = np.sqrt((kappa - 1) * (kappa + 1))
    # The reciprocal avoids subtracting two large, nearly equal numbers.
    z = 1 / (kappa + root)
    dkappa = np.array(
        [(1 - qnd**2) / (2 * qnd), ((1 - signal) - (1 + signal) / qnd**2) / 2]
    )
    dz = -z / root * dkappa
    dqnd = np.array([0.0, 1.0])
    # Expand in qnd-z so the polynomial denominator does not cancel near QND=1.
    h, w = (1 - qnd) * (1 + qnd), qnd - z
    dw = dqnd - dz
    a = h * (1 + qnd**2) + qnd * (5 * qnd**2 - 1) * w - 4 * qnd**2 * w**2
    da = (-4 * qnd**3 + (15 * qnd**2 - 1) * w - 8 * qnd * w**2) * dqnd + (
        qnd * (5 * qnd**2 - 1) - 8 * qnd**2 * w
    ) * dw
    c = h**2 + 6 * qnd * h * w + (9 * qnd**2 - 5) * w**2 - 4 * qnd * w**3
    dc = (
        -4 * qnd * h + 6 * (1 - 3 * qnd**2) * w + 18 * qnd * w**2 - 4 * w**3
    ) * dqnd + (6 * qnd * h + 2 * (9 * qnd**2 - 5) * w - 12 * qnd * w**2) * dw
    information = 4 * (qnd - z) ** 2 * a / ((1 - qnd**2) * (1 - qnd * z) * c)
    gradient = information * (
        2 * (dqnd - dz) / (qnd - z)
        + da / a
        + 2 * qnd * dqnd / (1 - qnd**2)
        + (z * dqnd + qnd * dz) / (1 - qnd * z)
        - dc / c
    )
    return float(information), np.asarray(gradient, dtype=np.float64)


def pair_correlation(model: ModelParameters, pair: NDArray[np.int64]) -> float:
    """QND η = R̃_aa + R̃_bb - 1 for two distinct state indices in pair.

    R̃ is the model transition's pair block, each row renormalised within the
    pair. The third state's landing therefore affects QND even at fixed λ.
    """
    keep, land = model.retention[pair], model.landing[pair]
    rows = keep + (1 - keep) * land.sum()
    return float(1 - ((1 - keep) * land[::-1] / rows).sum())


def pair_information_gradient(
    coordinates: FloatArray, pair: NDArray[np.int64]
) -> FloatArray:
    """Analytic (12,) gradient of pair_information in estimator free coordinates.

    Hold pair's two state indices fixed. Propagate q=d²/(4σ²) through the
    centre/noise tangents and QND through both retention and landing tangents,
    including the coordinate map's clipping and floor conventions.
    """
    model = physical_coordinates(coordinates)
    tangent = physical_coordinate_jacobian(coordinates)
    first, second = pair
    delta = model.centres[second] - model.centres[first]
    q = float(delta @ delta / (4 * model.variance))
    dq = (tangent.centres[:, second] - tangent.centres[:, first]) @ delta / (
        2 * model.variance
    ) - q * tangent.variance / model.variance
    gradient = all_lag_information_gradient(q, pair_correlation(model, pair))
    return gradient[0] * dq + gradient[1] * pair_correlation_gradient(coordinates, pair)


def pair_correlation_gradient(
    coordinates: FloatArray, pair: NDArray[np.int64]
) -> FloatArray:
    """Analytic (12,) gradient of pair_correlation in estimator free coordinates.

    Hold the two distinct pair indices fixed. Include the retention and landing
    tangents through the row renormalisation, including clipping/floor kinks.
    """
    model = physical_coordinates(coordinates)
    tangent = physical_coordinate_jacobian(coordinates)
    keep, land = model.retention[pair], model.landing[pair]
    dkeep, dland = tangent.retention[:, pair], tangent.landing[:, pair]
    rows = keep + (1 - keep) * land.sum()
    off_diagonal = (1 - keep) * land[::-1]
    doff = -land[::-1] * dkeep + (1 - keep) * dland[:, ::-1]
    drows = (1 - land.sum()) * dkeep + (1 - keep) * dland.sum(axis=1, keepdims=True)
    return -((doff * rows - off_diagonal * drows) / rows**2).sum(axis=1)


def pair_information(model: ModelParameters, pair: NDArray[np.int64]) -> float:
    """All-lag weak-drive information per read for a fixed pair of state indices.

    Use an infinite symmetric ...DRDRDR... chain with equal pair populations,
    q=d²/(4σ²) and QND from the row-renormalised pair transition block. Combine
    all lag j>=1 read products with exact long-run covariance and GLS weights.
    This defines a symmetric target, not the unequal-population chain's Fisher
    information. Larger dimensionless values mean more information.
    """
    delta = model.centres[pair[1]] - model.centres[pair[0]]
    return all_lag_information(
        float(delta @ delta / (4 * model.variance)), pair_correlation(model, pair)
    )
