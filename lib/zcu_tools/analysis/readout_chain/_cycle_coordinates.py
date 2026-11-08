"""Simplex coordinates, the nuisance layout and the affine-row chain rule."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ._cycle_emission import simplex, simplex_coordinates, simplex_gradient
from ._cycle_engine import mean_jacobian

if TYPE_CHECKING:
    from .model import Curve, CycleModel, CycleNuisances

# The 26 nuisance coordinates that follow the curve parameters. A simplex is an
# (s, f) pair with probabilities (s*f, s*(1-f), 1-s); see simplex.
ENDPOINTS = slice(0, 12)  # by state, then endpoint 0/1, then (s, f)
CENTERS = slice(12, 18)  # (I, Q) per state
VARIANCE = 18
THERMAL = slice(19, 21)  # (s, f)
RETENTION = slice(21, 24)  # one per state
LANDING = slice(24, 26)  # (s, f)
NUISANCE_SIZE = 26
# Centers through landing are shared by every point; the emission and moment
# kernels take these 14 coordinates as one row, in this order.
SHARED = slice(CENTERS.start, NUISANCE_SIZE)

# A per-point table row holds one free (s, f) drive pair per state, then the
# shared coordinates.
TABLE_DRIVE = slice(0, 6)
TABLE_SIZE = TABLE_DRIVE.stop + SHARED.stop - SHARED.start

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


def table_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Bounds of a per-point table row: drive pairs in [0, 1], shared as above."""
    lo, hi = nuisance_bounds()
    drive = TABLE_DRIVE.stop - TABLE_DRIVE.start
    return np.r_[np.zeros(drive), lo[SHARED]], np.r_[np.ones(drive), hi[SHARED]]


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
    df = curve_derivative(
        model.curve, axis, model.parameter_names, q[:n], np.asarray(model.bounds)
    )
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
    if set(parameters) != set(model.parameter_names):
        raise ValueError("parameters must name every curve parameter exactly once")
    theta = np.array([parameters[k] for k in model.parameter_names], dtype=float)
    bounds = np.asarray(model.bounds)
    if not np.isfinite(theta).all() or np.any(
        (theta < bounds[:, 0]) | (theta > bounds[:, 1])
    ):
        raise ValueError("curve parameter is not finite or outside bounds")
    return theta
