"""Simplex coordinates of the nuisance vector and the affine-row chain rule."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ._cycle_emission import simplex, simplex_coordinates, simplex_gradient
from ._cycle_engine import mean_jacobian
from ._cycle_layout import (
    CENTERS,
    ENDPOINTS,
    LANDING,
    NUISANCE_SIZE,
    RETENTION,
    SHARED,
    TABLE_DRIVE,
    THERMAL,
    VARIANCE,
)

if TYPE_CHECKING:
    from .model import Curve, CycleModel, CycleNuisances

NUISANCE_NAMES = tuple(
    [f"endpoint_{i}_{e}_{c}" for i in range(3) for e in range(2) for c in ("s", "f")]
    + [f"center_{i}_{c}" for i in range(3) for c in ("I", "Q")]
    + ["variance", "thermal_s", "thermal_f"]
    + [f"retention_{i}" for i in range(3)]
    + ["landing_s", "landing_f"]
)


def endpoints_of(nuisance: np.ndarray) -> np.ndarray:
    """Return the (3, 2, 3) drive endpoint probabilities of a nuisance vector."""
    pairs = nuisance[ENDPOINTS].reshape(6, 2)
    return np.array([simplex(*pair) for pair in pairs]).reshape(3, 2, 3)


def pack_nuisance(nuisance: "CycleNuisances") -> np.ndarray:
    """Return the 26 nuisance coordinates of nuisance, in the layout above."""
    packed = np.empty(NUISANCE_SIZE)
    packed[ENDPOINTS] = np.array(
        [simplex_coordinates(p) for p in nuisance.drive_endpoints.reshape(6, 3)]
    ).ravel()
    packed[CENTERS] = nuisance.centers.ravel()
    packed[VARIANCE] = nuisance.variance
    packed[THERMAL] = simplex_coordinates(nuisance.thermal)
    packed[RETENTION] = nuisance.read_retention
    packed[LANDING] = simplex_coordinates(nuisance.read_landing)
    return packed


def nuisance_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(NUISANCE_SIZE)
    hi = np.ones(NUISANCE_SIZE)
    lo[CENTERS], hi[CENTERS] = -np.inf, np.inf
    lo[VARIANCE], hi[VARIANCE] = np.finfo(float).tiny, np.inf
    # Exclude exactly deterministic off-diagonal reads, whose tail derivative
    # diverges. One machine epsilon is numerical boundary protection, not a prior.
    lo[LANDING] = np.finfo(float).eps
    hi[LANDING] = 1 - np.finfo(float).eps
    return lo, hi


def curve_values(
    curve: "Curve", axis: np.ndarray, names: tuple[str, ...], theta: np.ndarray
) -> np.ndarray:
    values = np.asarray(curve(axis, dict(zip(names, theta, strict=True))), dtype=float)
    if values.shape != (len(axis), 3):
        raise ValueError("curve must return shape (points, 3)")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("curve values must be finite and in [0, 1]")
    return values


def curve_derivative(
    derivative: "Curve",
    axis: np.ndarray,
    names: tuple[str, ...],
    theta: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        derivative(axis, dict(zip(names, theta, strict=True))), dtype=float
    )
    if values.shape != (len(axis), 3, len(names)) or not np.isfinite(values).all():
        raise ValueError(
            "curve derivative must be finite with shape (points, 3, parameters)"
        )
    return values


def joint_table(
    model: "CycleModel", axis: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-point drive matrices (F,3,3) and shared rows (F,14) at q.

    q holds the curve parameters, then the 26 nuisance coordinates.
    """
    n = len(model.parameter_names)
    fx = curve_values(model.curve, axis, model.parameter_names, q[:n])
    nuisance = q[n:]
    endpoints = endpoints_of(nuisance)
    drive = (
        endpoints[:, 0][None] * (1 - fx[:, :, None])
        + endpoints[:, 1][None] * fx[:, :, None]
    )
    rows = np.tile(nuisance[SHARED], (len(axis), 1))
    return drive, rows


def joint_mean_jacobian(
    model: "CycleModel", axis: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return means (F,65) and their Jacobian (F,65,len(q)) at q."""
    drive, rows = joint_table(model, axis, q)
    means, local = mean_jacobian(drive, rows)
    n = len(model.parameter_names)
    fx = curve_values(model.curve, axis, model.parameter_names, q[:n])
    df = curve_derivative(model.derivative, axis, model.parameter_names, q[:n])
    ends = endpoints_of(q[n:])
    jac = np.zeros((len(axis), means.shape[1], len(q)))
    for state in range(3):
        block = local[:, :, 2 * state : 2 * state + 2]
        slope = ends[state, 1, :2] - ends[state, 0, :2]
        jac[:, :, :n] += (block @ slope)[:, :, None] * df[:, state, None, :]
        for endpoint in range(2):
            offset = n + ENDPOINTS.start + 4 * state + 2 * endpoint
            grad = simplex_gradient(*q[offset : offset + 2])[:2]
            weight = fx[:, state] if endpoint else 1 - fx[:, state]
            jac[:, :, offset : offset + 2] = (block @ grad) * weight[:, None, None]
    jac[:, :, n + SHARED.start :] = local[:, :, TABLE_DRIVE.stop :]
    return means, jac


def parameter_vector(
    model: "CycleModel", parameters: Mapping[str, float]
) -> np.ndarray:
    """Return the curve parameter values as an array in parameter_names order.

    parameters must name every curve parameter of model exactly once, with
    finite values inside model.bounds; otherwise raise ValueError.
    """
    if set(parameters) != set(model.parameter_names):
        raise ValueError("parameters must name every curve parameter exactly once")
    theta = np.array([parameters[k] for k in model.parameter_names], dtype=float)
    bounds = np.asarray(model.bounds)
    if not np.isfinite(theta).all() or np.any(
        (theta < bounds[:, 0]) | (theta > bounds[:, 1])
    ):
        raise ValueError("curve parameter is not finite or outside bounds")
    return theta
