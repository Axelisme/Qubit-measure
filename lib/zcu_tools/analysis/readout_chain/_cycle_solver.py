"""Fixed-weight bounded GMM kernels and moment-derived state assignments."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from ._cycle_coordinates import (
    endpoints_of,
    joint_mean_jacobian,
    joint_table,
    nuisance_bounds,
)
from ._cycle_emission import simplex, simplex_coordinates
from ._cycle_engine import covariance, mean_jacobian
from ._cycle_layout import (
    CENTERS,
    ENDPOINTS,
    LANDING,
    NUISANCE_SIZE,
    RETENTION,
    THERMAL,
    VARIANCE,
)

if TYPE_CHECKING:
    from .model import CycleModel

_INFORMATION_RCOND = 1e-12
_ASSIGNMENT_EVALUATIONS = 200
_JOINT_EVALUATIONS = 1_000


@dataclass
class ScaledInformation:
    """Generalized covariance and null projection in column-normalized units.

    Keep the null projection separate from covariance: its zero information
    cannot be interpreted as zero uncertainty, and derived quantities can be
    estimable even when their individual coordinate components are not.
    """

    covariance: np.ndarray
    scale: np.ndarray
    null_projection: np.ndarray

    def errors(self, gradients: np.ndarray) -> np.ndarray:
        """Propagate rows of gradients, returning NaN for non-estimable rows.

        gradients has shape (..., quantities, parameters), with leading axes
        broadcast against this information's optional batch axes. Test null
        overlap in normalized units using the same 1e-6 singular-value cutoff
        as the 1e-12 information cutoff.
        """
        normalized = gradients / self.scale[..., None, :]
        overlap = np.linalg.norm(normalized @ self.null_projection, axis=-1)
        estimable = overlap**2 <= _INFORMATION_RCOND * np.sum(normalized**2, axis=-1)
        variance = np.einsum(
            "...ki,...ij,...kj->...k", gradients, self.covariance, gradients
        )
        return np.where(estimable, np.sqrt(np.maximum(0.0, variance)), np.nan)


@dataclass
class JointSolution:
    q: np.ndarray
    predicted_means: np.ndarray
    residual: np.ndarray
    information: ScaledInformation


def apply_weights(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Multiply each point's (65, ...) values by its (65, 65) weight matrix."""
    return np.einsum("nij,nj...->ni...", weights, values)


def scaled_information(weighted: np.ndarray) -> ScaledInformation:
    """Invert column-normalized information and retain its numerical null space.

    weighted has shape (observations, parameters). Apply the same 1e-12 cutoff
    to the normal-matrix inverse and the null space, then restore covariance units.
    """
    scale, normal = _scaled_normal_matrix(weighted)
    values, vectors = np.linalg.eigh(normal)
    null = np.abs(values) <= _INFORMATION_RCOND * np.max(
        np.abs(values), axis=-1, keepdims=True
    )
    projection = (vectors * null[..., None, :]) @ vectors.swapaxes(-1, -2)
    return ScaledInformation(_covariance_from_normal(scale, normal), scale, projection)


def _scaled_normal_matrix(weighted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the normal matrix after scaling each Jacobian column to unit norm."""
    scale = np.linalg.norm(weighted, axis=-2)
    scale = np.where(scale == 0, 1.0, scale)
    scaled = weighted / scale[..., None, :]
    return scale, scaled.swapaxes(-1, -2) @ scaled


def _covariance_from_normal(scale: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Restore the parameter units of the generalized inverse."""
    inverse = np.linalg.pinv(normal, rcond=_INFORMATION_RCOND, hermitian=True)
    return inverse / (scale[..., :, None] * scale[..., None, :])


def model_weights(
    drive: np.ndarray, rows: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    # The weights need the means; the kernel computes the Jacobian alongside them.
    means, _ = mean_jacobian(drive, rows)
    omega = covariance(drive, rows, means)
    lower = np.linalg.cholesky(omega)
    return np.linalg.inv(lower) * np.sqrt(counts[:, None, None])


def starting_nuisance(means: np.ndarray) -> np.ndarray:
    """Seed the 26 nuisance coordinates from the per-point feature means.

    Place two centers on either side of the mean IQ of all reads, along the
    principal axis of the first read's mean IQ across points, and the third
    center perpendicular to that axis; take the variance from the first
    read's same-read second moments. State labels are arbitrary here. Drive
    endpoints, thermal, retention and landing start at fixed interior values.
    No curve parameter is searched.
    """
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
    seed = np.empty(NUISANCE_SIZE)
    seed[ENDPOINTS] = np.array(
        [simplex_coordinates(p) for p in endpoints.reshape(6, 3)]
    ).ravel()
    seed[CENTERS] = centers.ravel()
    seed[VARIANCE] = variance
    seed[THERMAL] = (0.95, 0.7)
    seed[RETENTION] = (0.95, 0.8, 0.95)
    seed[LANDING] = (0.9, 0.7)
    return seed


def solve_joint(
    model: "CycleModel",
    axis: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    q: np.ndarray,
) -> JointSolution:
    """Fit the curve parameters and the 26 nuisance coordinates jointly.

    model supplies the per-state curve and its analytic derivative. axis has
    shape (F,), means (F,65) per-cycle feature means, counts (F,) cycle counts.
    q is the start: the curve parameters, then the nuisance layout of
    _cycle_layout. With curve parameters held at q, fit the nuisances for each
    of the six assignments of geometric states to curve rows; run two joint
    model-covariance weight stages from the lowest-objective assignment. Return
    the estimate, its normalized information, predicted means and last-stage
    whitened residual without changing the inputs. Raise RuntimeError if a joint
    stage does not converge; numpy.linalg.LinAlgError propagates from a model
    covariance that is not positive definite.
    """
    # Use one common metric so assignment scores are directly comparable.
    # Curves remain at caller init while the six nuisance seeds are fitted.
    weights = model_weights(*joint_table(model, axis, q), counts)
    candidates = [
        _joint_stage(
            model, axis, means, weights, assigned_seed(q, order), nuisance_only=True
        )
        for order in permutations(range(3))
    ]
    q = min(
        candidates,
        key=lambda seed: np.sum(
            apply_weights(weights, means - joint_mean_jacobian(model, axis, seed)[0])
            ** 2
        ),
    )
    for _ in range(2):
        weights = model_weights(*joint_table(model, axis, q), counts)
        q = _joint_stage(model, axis, means, weights, q)
    predicted, jac = joint_mean_jacobian(model, axis, q)
    weighted = apply_weights(weights, jac).reshape(-1, len(q))
    information = scaled_information(weighted)
    residual = apply_weights(weights, means - predicted)
    return JointSolution(q, predicted, residual, information)


def assigned_seed(q: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Move geometric states together, while leaving the model's curve rows fixed."""
    n = len(q) - NUISANCE_SIZE
    nuisance = q[n:]
    indices = np.array(order)
    endpoints = endpoints_of(nuisance)[indices][:, :, indices]
    assigned = np.empty(NUISANCE_SIZE)
    assigned[ENDPOINTS] = np.array(
        [simplex_coordinates(p) for p in endpoints.reshape(6, 3)]
    ).ravel()
    assigned[CENTERS] = nuisance[CENTERS].reshape(3, 2)[indices].ravel()
    assigned[VARIANCE] = nuisance[VARIANCE]
    assigned[THERMAL] = simplex_coordinates(simplex(*nuisance[THERMAL])[indices])
    assigned[RETENTION] = nuisance[RETENTION][indices]
    assigned[LANDING] = simplex_coordinates(simplex(*nuisance[LANDING])[indices])
    return np.r_[q[:n], assigned]


def _joint_stage(
    model: "CycleModel",
    axis: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    q: np.ndarray,
    *,
    nuisance_only: bool = False,
) -> np.ndarray:
    lo, hi = nuisance_bounds()
    limits = np.asarray(model.bounds)
    lo, hi = np.r_[limits[:, 0], lo], np.r_[limits[:, 1], hi]
    fixed = len(model.parameter_names) if nuisance_only else 0
    previous = np.full_like(q, np.nan)
    cached_mean = np.empty_like(means)
    cached_jac = np.empty((len(axis), 65, len(q)))

    def evaluate(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal previous, cached_mean, cached_jac
        if not np.array_equal(previous, v):
            cached_mean, cached_jac = joint_mean_jacobian(model, axis, v)
            previous = v.copy()
        return cached_mean, cached_jac

    def residual(v: np.ndarray) -> np.ndarray:
        full = np.r_[q[:fixed], v]
        return apply_weights(weights, means - evaluate(full)[0]).ravel()

    def derivative(v: np.ndarray) -> np.ndarray:
        full = np.r_[q[:fixed], v]
        return -apply_weights(weights, evaluate(full)[1][:, :, fixed:]).reshape(
            -1, len(v)
        )

    result = bounded_least_squares(
        residual,
        q[fixed:],
        derivative,
        (lo[fixed:], hi[fixed:]),
        nuisance_only=nuisance_only,
    )
    # Assignment scoring needs a feasible seed, not a converged nuisance fit.
    # Keep its last iterate at the budget cap; joint stages must still converge.
    if not result.success and not nuisance_only:
        raise RuntimeError(f"joint cycle fit did not converge: {result.message}")
    return np.r_[q[:fixed], result.x]


def bounded_least_squares(
    fun: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    jac: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    nuisance_only: bool,
) -> OptimizeResult:
    """Minimize the sum of squared residuals within coordinate bounds using SciPy.

    fun maps a parameter vector of shape (n,) to residuals of shape (m,).
    start is a finite initial parameter vector of shape (n,) within bounds.
    jac maps the same parameter vector to the residual Jacobian of shape
    (m, n). bounds contains lower and upper arrays of shape (n,), with each
    lower bound strictly below its upper bound; use infinities for unbounded
    coordinates. nuisance_only selects the smaller evaluation budget for
    state-assignment seeding instead of the joint-stage budget.

    SciPy uses x_scale="jac", trust-region reflective steps and 1e-9 for
    ftol, xtol and gtol.

    Return SciPy's OptimizeResult with the final x (n,), fun (m,), jac
    (m, n), cost, status, success and message. If SciPy stops because the
    evaluation budget is exhausted, it returns status=0 and success=False
    rather than raising here. Callers must inspect the result to decide
    whether to accept it. SciPy input-validation errors (such as ValueError
    for invalid bounds or nonfinite initial residuals) and exceptions from
    callbacks or numerical operations (including numpy.linalg.LinAlgError)
    propagate without being caught.
    """
    # Without installed scipy stubs pyright infers jac: str from the '2-point' default.
    analytic: Any = jac
    return least_squares(
        fun,
        start,
        jac=analytic,
        bounds=bounds,
        x_scale="jac",
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        max_nfev=_ASSIGNMENT_EVALUATIONS if nuisance_only else _JOINT_EVALUATIONS,
    )
