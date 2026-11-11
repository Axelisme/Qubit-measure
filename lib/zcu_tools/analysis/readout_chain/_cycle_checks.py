"""Nested fixed-weight fits with batched pointwise Gauss-Newton elimination."""

from dataclasses import dataclass

import numpy as np

from ._cycle_emission import simplex, simplex_gradient
from ._cycle_layout import (
    ROW_SIZE,
    ROW_THERMAL,
    TABLE_DRIVE,
    TABLE_SHARED,
    TABLE_THERMAL,
)
from ._cycle_solver import (
    ScaledInformation,
    apply_weights,
    bounded_least_squares,
    local_drive,
    local_mean_jacobian,
    scaled_covariance,
    scaled_information,
)

# Lower and upper bounds of the table columns being fitted.
Limits = tuple[np.ndarray, np.ndarray]
# Table, predicted means, Jacobian, weighted residual and per-point cost.
LineState = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class InnerSolution:
    """Pointwise fit and its explicit optimality/failure outcome.

    table: Fitted table, shape (F,20).
    residual: Final weighted residual, shape (F,65).
    converged: Whether every point meets bound-scaled first-order optimality.
    reason: Optimality reached, iteration budget exhausted, or line search failed.
    """

    table: np.ndarray
    residual: np.ndarray
    converged: bool
    reason: str


@dataclass
class SharedSolution:
    """Fit of the shared coordinates with free drive pairs at each point.

    table: Shape (F,20), fitted table rows in the layout of _cycle_layout.
    J: Weighted residual sum of squares at table, dimensionless.
    covariance_shared: Shape (14,14), covariance of the shared coordinates.
    covariance_local: Shape (F,6,6), per-point covariance of the drive pairs at
        fixed shared coordinates, with column-normalized inverse information.
    coupling: Shape (F,6,14), per-point covariance_local times the local-shared
        normal block. A change of the shared coordinates moves the fitted drive
        pairs by minus coupling times that change.
    information_local: Local covariance and normalized null projection.
    information_shared: Schur-complement covariance and normalized null projection.
    """

    table: np.ndarray
    J: float
    covariance_shared: np.ndarray
    covariance_local: np.ndarray
    coupling: np.ndarray
    information_local: ScaledInformation
    information_shared: ScaledInformation


def _normal(jac: np.ndarray, held: np.ndarray) -> np.ndarray:
    normal = np.einsum("nfi,nfj->nij", jac, jac)
    # Numerical damping only affects iteration, not the reported covariance.
    normal += (1e-10 * np.trace(normal, axis1=1, axis2=2) + 1e-14)[
        :, None, None
    ] * np.eye(jac.shape[2])
    normal = np.where(held[:, :, None] | held[:, None, :], 0.0, normal)
    return normal + held[:, :, None] * np.eye(jac.shape[2])


def _step(
    table: np.ndarray,
    cols: np.ndarray,
    jac: np.ndarray,
    residual: np.ndarray,
    limits: Limits,
) -> np.ndarray:
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


def inner(
    table: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    cols: np.ndarray,
    limits: Limits,
    steps: int = 40,
) -> InnerSolution:
    """Fit the table columns cols independently at each point; hold the others.

    table has shape (F,20) in the layout of _cycle_layout, means (F,65), weights
    (F,65,65) whitening matrices. cols indexes the fitted columns of table and
    limits gives their lower and upper bounds. Run bounded damped Gauss-Newton
    with a per-point halving line search for at most steps iterations. Require
    bound-scaled first-order optimality below 1e-9, the joint/shared solver gtol:
    multiply each residual-gradient coordinate by the distance to its downhill
    bound (one for unbounded directions). Zero motion is not convergence.
    Return an InnerSolution; budget exhaustion or failed descent is explicit.
    The inputs are not changed; numpy.linalg.LinAlgError propagates from a
    singular solve.
    """
    table = table.copy()
    predicted, jac = local_mean_jacobian(table)
    residual = apply_weights(weights, means - predicted)
    cost = np.sum(residual**2, axis=1)
    for iteration in range(steps + 1):
        weighted = apply_weights(weights, jac[:, :, cols])
        optimal = _optimality(table[:, cols], weighted, residual, limits) < 1e-9
        if optimal.all():
            return InnerSolution(table, residual, True, "optimality reached")
        if iteration == steps:
            break
        delta = _step(table, cols, weighted, residual, limits)
        previous_cost = cost
        table, predicted, jac, residual, cost = _line_search(
            (table, predicted, jac, residual, cost), delta, cols, means, weights, limits
        )
        stalled = (cost >= previous_cost) & ~optimal
        if stalled.any():
            # A roundoff-sized move can reach optimality without lowering J.
            weighted = apply_weights(weights, jac[:, :, cols])
            optimal = _optimality(table[:, cols], weighted, residual, limits) < 1e-9
            if np.any(stalled & ~optimal):
                return InnerSolution(
                    table, residual, False, "line search found no descent"
                )
    return InnerSolution(table, residual, False, "iteration budget exhausted")


def _optimality(
    values: np.ndarray, jac: np.ndarray, residual: np.ndarray, limits: Limits
) -> np.ndarray:
    lo, hi = limits
    # residual = observed - predicted, so positive gradient points downhill.
    gradient = np.einsum("nfi,nf->ni", jac, residual)
    distance = np.where(gradient > 0, hi - values, values - lo)
    distance = np.where(np.isfinite(distance), distance, 1.0)
    return np.max(np.abs(gradient * distance), axis=1)


def _line_search(
    state: LineState,
    delta: np.ndarray,
    cols: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    limits: Limits,
) -> LineState:
    table, predicted, jac, residual, cost = state
    lo, hi = limits
    accept = np.zeros(len(table), dtype=bool)
    candidate, pred, derivative, r, c = table, predicted, jac, residual, cost
    for _ in range(12):
        candidate = table.copy()
        candidate[:, cols] = np.clip(table[:, cols] + delta, lo, hi)
        pred, derivative = local_mean_jacobian(candidate)
        r = apply_weights(weights, means - pred)
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
    limits: Limits,
) -> SharedSolution:
    """Fit the 14 shared coordinates with the drive pairs eliminated at each point.

    table is the (F,20) starting table in the layout of _cycle_layout; each
    point's drive pairs and the first row's shared columns seed the fit. means
    has shape (F,65), weights (F,65,65) fixed whitening matrices, limits the
    lower and upper bounds of all 20 columns. Each least-squares residual first
    refits every point's drive pairs with inner. Return the SharedSolution at
    the optimum without changing the inputs. Raise RuntimeError if the shared
    fit does not converge.
    """
    lo, hi = limits
    cols = np.arange(TABLE_DRIVE.stop)
    current = table.copy()
    residual = np.zeros_like(means)

    def fun(shared: np.ndarray) -> np.ndarray:
        nonlocal current, residual
        current[:, TABLE_SHARED] = shared
        fitted = inner(
            current, means, weights, cols, (lo[TABLE_DRIVE], hi[TABLE_DRIVE])
        )
        if not fitted.converged:
            raise RuntimeError(f"sharing cycle check did not converge: {fitted.reason}")
        current, residual = fitted.table, fitted.residual
        return residual.ravel()

    def jacobian(shared: np.ndarray) -> np.ndarray:
        # least_squares' trf calls jac only at the accepted point, right after fun
        # evaluated it, so current still holds the drive pairs solved for this shared.
        current[:, TABLE_SHARED] = shared
        jac = apply_weights(weights, local_mean_jacobian(current)[1])
        local, global_ = jac[:, :, TABLE_DRIVE], jac[:, :, TABLE_SHARED]
        coupling = np.einsum("nfi,nfj->nij", local, global_)
        projected = global_ - local @ (scaled_covariance(local) @ coupling)
        return -projected.reshape(-1, ROW_SIZE)

    result = bounded_least_squares(
        fun,
        current[0, TABLE_SHARED],
        jacobian,
        (lo[TABLE_SHARED], hi[TABLE_SHARED]),
        100,
    )
    if not result.success:
        raise RuntimeError(f"shared cycle check did not converge: {result.message}")
    fun(result.x)
    jac = apply_weights(weights, local_mean_jacobian(current)[1])
    local, global_ = jac[:, :, TABLE_DRIVE], jac[:, :, TABLE_SHARED]
    # Normal-matrix blocks of the drive pairs (local) and shared coordinates.
    local_shared = np.einsum("nfi,nfj->nij", local, global_)
    information_local = scaled_information(local)
    covariance_local = information_local.covariance
    coupling = covariance_local @ local_shared
    # Form the Schur complement as a projected Jacobian to retain column scales
    # and avoid subtracting nearly equal normal-matrix blocks.
    projected = global_ - local @ coupling
    information_shared = scaled_information(projected.reshape(-1, ROW_SIZE))
    covariance_shared = information_shared.covariance
    return SharedSolution(
        current,
        float(np.sum(residual**2)),
        covariance_shared,
        covariance_local,
        coupling,
        information_local,
        information_shared,
    )


def populations(solution: SharedSolution) -> tuple[np.ndarray, np.ndarray]:
    """Return (F,3) populations and errors from the shared-nuisance fit.

    Propagate each population gradient through the local and Schur-corrected
    shared information separately. An error is NaN if either gradient involves
    that information's null space; unrelated nuisance rank loss does not remove
    errors for estimable populations.
    """
    table = solution.table
    drive = local_drive(table)
    pi = np.array([simplex(*r[TABLE_THERMAL]) for r in table])
    values = np.einsum("ni,nij->nj", pi, drive)
    local = np.zeros((len(table), 3, TABLE_DRIVE.stop))
    shared = np.zeros((len(table), 3, ROW_SIZE))
    for x, row in enumerate(table):
        for state in range(3):
            local[x, :, 2 * state : 2 * state + 2] = pi[x, state] * simplex_gradient(
                *row[2 * state : 2 * state + 2]
            )
        shared[x, :, ROW_THERMAL] = drive[x].T @ simplex_gradient(*row[TABLE_THERMAL])
    corrected = shared - local @ solution.coupling
    local_errors = solution.information_local.errors(local)
    shared_errors = solution.information_shared.errors(corrected)
    return values, np.sqrt(local_errors**2 + shared_errors**2)
