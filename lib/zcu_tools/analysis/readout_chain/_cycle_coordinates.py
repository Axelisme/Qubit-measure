"""Simplex coordinates and the affine-row chain rule."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ._cycle_emission import simplex, simplex_coordinates, simplex_gradient
from ._cycle_engine import mean_jacobian

if TYPE_CHECKING:
    from .model import Curve

NUISANCE_NAMES = tuple(
    [f"endpoint_{i}_{e}_{c}" for i in range(3) for e in range(2) for c in ("s", "f")]
    + [f"center_{i}_{c}" for i in range(3) for c in ("I", "Q")]
    + ["variance", "thermal_s", "thermal_f"]
    + [f"retention_{i}" for i in range(3)]
    + ["landing_s", "landing_f"]
)


def endpoints_of(q: np.ndarray) -> np.ndarray:
    return np.array([simplex(*v) for v in q[:12].reshape(6, 2)]).reshape(3, 2, 3)


def pack_nuisance(nuisance) -> np.ndarray:
    return np.concatenate(
        [
            np.array(
                [simplex_coordinates(p) for p in nuisance.drive_endpoints.reshape(6, 3)]
            ).ravel(),
            nuisance.centers.ravel(),
            [nuisance.variance],
            simplex_coordinates(nuisance.thermal),
            nuisance.read_retention,
            simplex_coordinates(nuisance.read_landing),
        ]
    )


def nuisance_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(26)
    hi = np.ones(26)
    lo[12:18], hi[12:19] = -np.inf, np.inf
    lo[18] = np.finfo(float).tiny
    # Exclude exactly deterministic off-diagonal reads, whose tail derivative
    # diverges. One machine epsilon is numerical boundary protection, not a prior.
    lo[24:26] = np.finfo(float).eps
    hi[24:26] = 1 - np.finfo(float).eps
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
    curve: "Curve",
    axis: np.ndarray,
    names: tuple[str, ...],
    theta: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    # Only the author's black-box curve is differenced; chain moments and
    # all readout/transition derivatives are analytic.
    cols = []
    for k, value in enumerate(theta):
        step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(value))
        left, right = theta.copy(), theta.copy()
        left[k] = max(bounds[k, 0], value - step)
        right[k] = min(bounds[k, 1], value + step)
        fp = curve_values(curve, axis, names, right)
        fm = curve_values(curve, axis, names, left)
        cols.append((fp - fm) / (right[k] - left[k]))
    return np.stack(cols, axis=2)


def joint_table(
    model, axis: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = len(model.parameter_names)
    fx = curve_values(model.curve, axis, model.parameter_names, q[:n])
    endpoints = endpoints_of(q[n:])
    drive = (
        endpoints[:, 0][None] * (1 - fx[:, :, None])
        + endpoints[:, 1][None] * fx[:, :, None]
    )
    rows = np.tile(q[n + 12 :], (len(axis), 1))
    return drive, rows


def joint_mean_jacobian(model, axis: np.ndarray, q: np.ndarray):
    drive, rows = joint_table(model, axis, q)
    means, local = mean_jacobian(drive, rows)
    n = len(model.parameter_names)
    fx = curve_values(model.curve, axis, model.parameter_names, q[:n])
    df = curve_derivative(
        model.curve, axis, model.parameter_names, q[:n], np.asarray(model.bounds)
    )
    ends = endpoints_of(q[n:])
    jac = np.zeros((len(axis), 65, len(q)))
    for state in range(3):
        block = local[:, :, 2 * state : 2 * state + 2]
        slope = ends[state, 1, :2] - ends[state, 0, :2]
        jac[:, :, :n] += (block @ slope)[:, :, None] * df[:, state, None, :]
        for endpoint in range(2):
            offset = n + 4 * state + 2 * endpoint
            grad = simplex_gradient(*q[offset : offset + 2])[:2]
            weight = fx[:, state] if endpoint else 1 - fx[:, state]
            jac[:, :, offset : offset + 2] = (block @ grad) * weight[:, None, None]
    jac[:, :, n + 12 :] = local[:, :, 6:]
    return means, jac


def parameter_vector(model, parameters: Mapping[str, float]) -> np.ndarray:
    if set(parameters) != set(model.parameter_names):
        raise ValueError("parameters must name every curve parameter exactly once")
    theta = np.array([parameters[k] for k in model.parameter_names], dtype=float)
    bounds = np.asarray(model.bounds)
    if not np.isfinite(theta).all() or np.any(
        (theta < bounds[:, 0]) | (theta > bounds[:, 1])
    ):
        raise ValueError("curve parameter is not finite or outside bounds")
    return theta
