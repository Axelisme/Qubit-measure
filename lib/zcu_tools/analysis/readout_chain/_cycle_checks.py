"""Nested fixed-weight fits with batched pointwise Gauss-Newton elimination."""

from dataclasses import dataclass

import numpy as np

from ._cycle_emission import simplex, simplex_gradient
from ._cycle_solver import (
    bounded_least_squares,
    local_drive,
    local_mean_jacobian,
    whiten,
)


@dataclass
class SharedSolution:
    table: np.ndarray
    J: float
    covariance_shared: np.ndarray
    covariance_local: np.ndarray
    coupling: np.ndarray


def _normal(jac: np.ndarray, held: np.ndarray) -> np.ndarray:
    normal = np.einsum("nfi,nfj->nij", jac, jac)
    # Numerical damping only affects iteration, not the reported covariance.
    normal += (1e-10 * np.trace(normal, axis1=1, axis2=2) + 1e-14)[
        :, None, None
    ] * np.eye(jac.shape[2])
    normal = np.where(held[:, :, None] | held[:, None, :], 0.0, normal)
    return normal + held[:, :, None] * np.eye(jac.shape[2])


def _step(table, cols, jac, residual, limits):
    lo, hi = limits
    values = table[:, cols]
    gradient = np.einsum("nfi,nf->ni", jac, residual)
    held = ((values <= lo) & (gradient < 0)) | ((values >= hi) & (gradient > 0))
    delta = np.zeros_like(values)
    for _ in range(len(cols) + 1):
        delta = np.linalg.solve(
            _normal(jac, held), np.where(held, 0.0, gradient)[:, :, None]
        )[:, :, 0]
        blocked = ~held & (
            ((values <= lo) & (delta < 0)) | ((values >= hi) & (delta > 0))
        )
        if not blocked.any():
            break
        held |= blocked
    with np.errstate(divide="ignore", invalid="ignore"):
        room = np.where(
            delta > 0,
            (hi - values) / delta,
            np.where(delta < 0, (lo - values) / delta, np.inf),
        )
    return delta * np.minimum(1.0, room.min(axis=1))[:, None]


def inner(table, means, weights, cols, limits, steps=40):
    table = table.copy()
    predicted, jac = local_mean_jacobian(table)
    residual = whiten(weights, means - predicted)
    cost = np.sum(residual**2, axis=1)
    for _ in range(steps):
        weighted = whiten(weights, jac[:, :, cols])
        delta = _step(table, cols, weighted, residual, limits)
        if np.max(np.abs(delta)) < 1e-9:
            break
        previous = table
        table, predicted, jac, residual, cost = _line_search(
            (table, predicted, jac, residual, cost), delta, cols, means, weights, limits
        )
        if np.max(np.abs(table[:, cols] - previous[:, cols])) < 1e-9:
            break
    return table, residual


def _line_search(state, delta, cols, means, weights, limits):
    table, predicted, jac, residual, cost = state
    lo, hi = limits
    accept = np.zeros(len(table), dtype=bool)
    candidate, pred, derivative, r, c = table, predicted, jac, residual, cost
    for _ in range(12):
        candidate = table.copy()
        candidate[:, cols] = np.clip(table[:, cols] + delta, lo, hi)
        pred, derivative = local_mean_jacobian(candidate)
        r = whiten(weights, means - pred)
        c = np.sum(r**2, axis=1)
        accept = c <= cost
        if accept.all():
            break
        delta = np.where(accept[:, None], delta, 0.5 * delta)
    return (
        np.where(accept[:, None], candidate, table),
        np.where(accept[:, None], pred, predicted),
        np.where(accept[:, None, None], derivative, jac),
        np.where(accept[:, None], r, residual),
        np.where(accept, c, cost),
    )


def solve_shared(
    table: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    limits: tuple[np.ndarray, np.ndarray],
) -> SharedSolution:
    lo, hi = limits
    cols = np.arange(6)
    current = table.copy()
    residual = np.zeros_like(means)

    def fun(shared):
        nonlocal current, residual
        current[:, 6:] = shared
        current, residual = inner(current, means, weights, cols, (lo[:6], hi[:6]))
        return residual.ravel()

    def jacobian(shared):
        current[:, 6:] = shared
        jac = whiten(weights, local_mean_jacobian(current)[1])
        local, global_ = jac[:, :, :6], jac[:, :, 6:]
        normal = np.einsum("nfi,nfj->nij", local, local)
        coupling = np.einsum("nfi,nfj->nij", local, global_)
        projected = global_ - local @ (
            np.linalg.pinv(normal, rcond=1e-12, hermitian=True) @ coupling
        )
        return -projected.reshape(-1, 14)

    result = bounded_least_squares(fun, current[0, 6:], jacobian, (lo[6:], hi[6:]), 100)
    if not result.success:
        raise RuntimeError(f"shared cycle check did not converge: {result.message}")
    fun(result.x)
    jac = whiten(weights, local_mean_jacobian(current)[1])
    local, global_ = jac[:, :, :6], jac[:, :, 6:]
    fvv = np.einsum("nfi,nfj->nij", local, local)
    fvs = np.einsum("nfi,nfj->nij", local, global_)
    cv = np.linalg.pinv(fvv, rcond=1e-12, hermitian=True)
    coupling = cv @ fvs
    schur = np.einsum("nfi,nfj->ij", global_, global_) - np.einsum(
        "nij,nik->jk", fvs, coupling
    )
    cs = np.linalg.pinv(schur, rcond=1e-12, hermitian=True)
    return SharedSolution(current, float(np.sum(residual**2)), cs, cv, coupling)


def populations(solution: SharedSolution) -> tuple[np.ndarray, np.ndarray]:
    table = solution.table
    drive = local_drive(table)
    pi = np.array([simplex(*r[13:15]) for r in table])
    values = np.einsum("ni,nij->nj", pi, drive)
    local = np.zeros((len(table), 3, 6))
    shared = np.zeros((len(table), 3, 14))
    for x, row in enumerate(table):
        for state in range(3):
            local[x, :, 2 * state : 2 * state + 2] = pi[x, state] * simplex_gradient(
                *row[2 * state : 2 * state + 2]
            )
        shared[x, :, 7:9] = drive[x].T @ simplex_gradient(*row[13:15])
    corrected = shared - local @ solution.coupling
    cov = local @ solution.covariance_local @ local.transpose(0, 2, 1)
    cov += corrected @ solution.covariance_shared @ corrected.transpose(0, 2, 1)
    return values, np.sqrt(np.maximum(0.0, np.diagonal(cov, axis1=1, axis2=2)))
