"""Nested fixed-weight fits with pointwise bounded least-squares elimination."""

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


@dataclass
class InnerSolution:
    """Pointwise fit and its explicit optimality/failure outcome.

    table: Fitted table, shape (F,20).
    residual: Final weighted residual, shape (F,65).
    converged: Whether every point meets primary or allowed stalled optimality.
    reason: Optimality reached, or the failed point and solver stop.
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


def inner(
    table: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    cols: np.ndarray,
    limits: Limits,
) -> InnerSolution:
    """Fit table columns cols independently at each point; hold the others.

    table has shape (F,20) in the layout of _cycle_layout, means (F,65), weights
    (F,65,65) whitening matrices. cols indexes the fitted columns and limits
    gives their lower and upper bounds. Project the start into these bounds.
    Each point uses bounded least squares with max_nfev=200, ftol=xtol=1e-12
    and gtol=1e-9. Independently check bound-scaled optimality <= 1e-9;
    only ftol/xtol stops may instead satisfy scale-invariant optimality <= 1e-6.
    Budget exhaustion always fails. Return the table and residual without
    changing inputs; failure names the point and solver stop.
    """
    table = table.copy()
    table[:, cols] = np.clip(table[:, cols], *limits)
    residual = np.zeros_like(means)
    for point in range(len(table)):
        fitted = _point(
            table[point : point + 1],
            means[point : point + 1],
            weights[point : point + 1],
            cols,
            limits,
        )
        table[point], residual[point] = fitted.table[0], fitted.residual[0]
        if not fitted.converged:
            return InnerSolution(
                table, residual, False, f"point {point}: {fitted.reason}"
            )
    return InnerSolution(table, residual, True, "optimality reached")


def _point(
    table: np.ndarray,
    means: np.ndarray,
    weights: np.ndarray,
    cols: np.ndarray,
    limits: Limits,
) -> InnerSolution:
    def fun(values: np.ndarray) -> np.ndarray:
        table[0, cols] = values
        predicted = local_mean_jacobian(table)[0]
        return apply_weights(weights, means - predicted).ravel()

    def jacobian(values: np.ndarray) -> np.ndarray:
        table[0, cols] = values
        jac = local_mean_jacobian(table)[1][:, :, cols]
        return -apply_weights(weights, jac)[0]

    # This is a function-evaluation budget, not a Gauss-Newton iteration count.
    # Tighten only nested point fits; joint/shared outer settings stay unchanged.
    result = bounded_least_squares(
        fun, table[0, cols], jacobian, limits, 200, stopping_tolerance=1e-12
    )
    residual = fun(result.x)[None, :]
    weighted = -jacobian(result.x)[None, :, :]
    optimal = _optimality(table[:, cols], weighted, residual, limits)[0] <= 1e-9
    # 1e-6 is about 75 times the measured roundoff stall maximum 1.34e-8,
    # but over four orders below the nonoptimal 0.0399. A small step or
    # SciPy success alone is not optimality; require this ratio at ftol/xtol.
    stalled_optimal = result.status in (2, 3, 4) and (
        _scale_invariant_optimality(weighted, residual, result.active_mask[None, :])[0]
        <= 1e-6
    )
    converged = result.status > 0 and result.nfev < 200 and (optimal or stalled_optimal)
    reason = (
        f"{result.message}; bound-scaled optimality not reached"
        if not converged
        else "optimality reached"
    )
    return InnerSolution(table, residual, bool(converged), reason)


def _optimality(
    values: np.ndarray, jac: np.ndarray, residual: np.ndarray, limits: Limits
) -> np.ndarray:
    lo, hi = limits
    # residual = observed - predicted, so positive gradient points downhill.
    gradient = np.einsum("nfi,nf->ni", jac, residual)
    distance = np.where(gradient > 0, hi - values, values - lo)
    distance = np.where(np.isfinite(distance), distance, 1.0)
    return np.max(np.abs(gradient * distance), axis=1)


def _scale_invariant_optimality(
    jac: np.ndarray, residual: np.ndarray, active_mask: np.ndarray
) -> np.ndarray:
    """Return max |projected A.T r| / (column norm of A * norm of r).

    A is the weighted prediction Jacobian and r is observed minus predicted,
    whitened by the same weights. SciPy's active_mask defines numerical bound
    contact (-1 lower, +1 upper). Zero only outward downhill directions;
    inward directions remain included. Guard zero denominators with float
    tiny; zero columns or residuals have zero gradient. Column/residual scaling
    cancel in the ratio.
    """
    gradient = np.einsum("nfi,nf->ni", jac, residual)
    outward = ((active_mask < 0) & (gradient < 0)) | (
        (active_mask > 0) & (gradient > 0)
    )
    projected = np.where(outward, 0.0, gradient)
    scale = np.linalg.norm(jac, axis=1) * np.linalg.norm(residual, axis=1)[:, None]
    return np.max(np.abs(projected) / np.maximum(scale, np.finfo(float).tiny), axis=1)


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
