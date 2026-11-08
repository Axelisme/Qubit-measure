"""Fixed-weight bounded GMM kernels and moment-derived single starts."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from ._cycle_coordinates import joint_mean_jacobian, joint_table, nuisance_bounds
from ._cycle_emission import simplex, simplex_coordinates, simplex_gradient
from ._cycle_engine import covariance, mean_jacobian


@dataclass
class JointSolution:
    q: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    J: float


def whiten(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj...->ni...", weights, values)


def model_weights(
    drive: np.ndarray, rows: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    means = mean_jacobian(drive, rows)[0]
    omega = covariance(drive, rows, means)
    lower = np.linalg.cholesky(omega)
    return np.linalg.inv(lower) * np.sqrt(counts[:, None, None])


def starting_nuisance(means: np.ndarray) -> np.ndarray:
    """Research's summary-only geometry seed, without its curve grid search."""
    first = means[:, :2]
    center = means[:, :32].reshape(-1, 16, 2).mean(axis=(0, 1))
    second = np.array(
        [
            [means[:, 32].mean(), means[:, 33].mean()],
            [means[:, 33].mean(), means[:, 34].mean()],
        ]
    )
    scatter = second - np.outer(first.mean(axis=0), first.mean(axis=0))
    direction = np.linalg.svd(first - first.mean(axis=0), full_matrices=False)[2][0]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    perpendicular = np.array([-direction[1], direction[0]])
    amplitude = max(
        np.ptp(first @ direction), 0.02 * np.sqrt(max(np.trace(scatter), 1e-8))
    )
    centers = np.array(
        [
            center - amplitude * direction,
            center + amplitude * direction,
            center + 1.5 * amplitude * perpendicular,
        ]
    )
    variance = max(1e-4, np.linalg.eigvalsh(scatter).min() - 0.25 * amplitude**2)
    endpoints = np.array(
        [
            [[0.8, 0.15, 0.05], [0.2, 0.65, 0.15]],
            [[0.65, 0.25, 0.1], [0.05, 0.9, 0.05]],
            [[0.05, 0.05, 0.9], [0.15, 0.15, 0.7]],
        ]
    )
    return np.concatenate(
        [
            np.array([simplex_coordinates(p) for p in endpoints.reshape(6, 3)]).ravel(),
            centers.ravel(),
            [variance, 0.95, 0.7, 0.95, 0.8, 0.95, 0.9, 0.7],
        ]
    )


def solve_joint(
    model, axis: np.ndarray, means: np.ndarray, counts: np.ndarray, q: np.ndarray
) -> JointSolution:
    lo, hi = nuisance_bounds()
    limits = np.asarray(model.bounds)
    lo, hi = np.r_[limits[:, 0], lo], np.r_[limits[:, 1], hi]
    weights = np.empty((len(axis), 65, 65))
    for _ in range(2):
        weights = model_weights(*joint_table(model, axis, q), counts)
        q = _joint_stage(model, axis, means, weights, q, (lo, hi))
    predicted, jac = joint_mean_jacobian(model, axis, q)
    weighted = whiten(weights, jac).reshape(-1, len(q))
    cov = np.linalg.pinv(weighted.T @ weighted, rcond=1e-12, hermitian=True)
    residual = whiten(weights, means - predicted)
    return JointSolution(q, cov, weights, float(np.sum(residual**2)))


def _joint_stage(model, axis, means, weights, q, bounds):
    previous = np.full_like(q, np.nan)
    cached_mean = np.empty_like(means)
    cached_jac = np.empty((len(axis), 65, len(q)))

    def evaluate(v):
        nonlocal previous, cached_mean, cached_jac
        if not np.array_equal(previous, v):
            cached_mean, cached_jac = joint_mean_jacobian(model, axis, v)
            previous = v.copy()
        return cached_mean, cached_jac

    def residual(v):
        return whiten(weights, means - evaluate(v)[0]).ravel()

    def derivative(v):
        return -whiten(weights, evaluate(v)[1]).reshape(-1, len(v))

    result = bounded_least_squares(residual, q, derivative, bounds, 200)
    if not result.success:
        raise RuntimeError(f"joint cycle fit did not converge: {result.message}")
    return result.x


def bounded_least_squares(
    fun: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    jac: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    budget: int,
) -> OptimizeResult:
    # SciPy has no installed stubs: its inferred jac='2-point' annotation omits
    # documented callable Jacobians. Keep that untyped backend at this boundary.
    backend: Callable[..., OptimizeResult] = least_squares
    return backend(
        fun,
        start,
        jac=jac,
        bounds=bounds,
        x_scale="jac",
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=budget,
    )


def local_table(drive: np.ndarray, rows: np.ndarray) -> np.ndarray:
    coordinates = np.array([[simplex_coordinates(p) for p in d] for d in drive])
    return np.c_[coordinates.reshape(len(drive), 6), rows]


def local_drive(table: np.ndarray) -> np.ndarray:
    return np.array([[simplex(*p) for p in t[:6].reshape(3, 2)] for t in table])


def local_mean_jacobian(table: np.ndarray):
    means, jac = mean_jacobian(local_drive(table), table[:, 6:])
    for state in range(3):
        block = slice(2 * state, 2 * state + 2)
        grad = np.array([simplex_gradient(*r[block])[:2] for r in table])
        jac[:, :, block] = jac[:, :, block] @ grad
    return means, jac
