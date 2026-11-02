"""Self-calibration of shared herald and signal IQ readout."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from zcu_tools.analysis.failure import AnalysisFailed


@dataclass(frozen=True)
class TransferResult:
    """Self-calibrated readout and state transfer, ordered by population.

    joint_probabilities: (N, n, n) joint herald/signal probabilities J.
    transition_matrices: (N, n, n) row-normalized K[i, j] = P(i -> j).
    change_probability: (N,) probability C = 1 - trace(J).
    populations: (n,) mean herald populations, in descending order.
    centers: (n,) complex state centers in the original IQ units.
    sigma: Gaussian standard deviation per quadrature, in original IQ units.
    singular_values: (n+1,) subspace singular values divided by the largest.
    Unmeasured points have NaN entries in J, K and C.
    """

    joint_probabilities: NDArray[np.float64]
    transition_matrices: NDArray[np.float64]
    change_probability: NDArray[np.float64]
    populations: NDArray[np.float64]
    centers: NDArray[np.complex128]
    sigma: float
    singular_values: NDArray[np.float64]


def selfcal_transfer(
    sums: NDArray[np.float64],
    n_states: int = 2,
    *,
    previous: TransferResult | None = None,
) -> TransferResult:
    """Infer shared readout calibration and transfer from (N, 6, 6) IQ sums."""
    if n_states not in (2, 3):
        raise ValueError("n_states must be 2 or 3")
    if sums.ndim != 3 or sums.shape[1:] != (6, 6):
        raise ValueError("sums must have shape (N, 6, 6)")
    measured = np.isfinite(sums).all(axis=(1, 2)) & (sums[:, 0, 0] > 0)
    if not measured.any():
        raise AnalysisFailed("No measured points for readout identification")
    moments, mean, scale = _normalized_moments(sums[measured])
    stack = np.concatenate([moments, moments.transpose(0, 2, 1)], axis=0)
    vectors, singular_values, _ = np.linalg.svd(
        stack.transpose(1, 0, 2).reshape(6, -1), full_matrices=False
    )
    if singular_values[n_states - 1] <= max(1e-8 * singular_values[0], 3 * singular_values[n_states]):
        raise AnalysisFailed("Requested states are not identifiable above the singular-value noise floor")
    projector = np.eye(6) - vectors[:, :n_states] @ vectors[:, :n_states].T
    centers, variance = _fit_centers(projector, n_states)
    inverse = np.linalg.pinv(_lift(centers, variance).T)
    joint = inverse @ moments @ inverse.T
    rows = joint.sum(axis=-1)
    populations = rows.mean(axis=0)
    order = np.argsort(-populations)
    joint = joint[:, order][:, :, order]
    all_joint = np.full((len(sums), n_states, n_states), np.nan)
    all_transition = np.full_like(all_joint, np.nan)
    all_change = np.full(len(sums), np.nan)
    all_joint[measured] = joint
    all_transition[measured] = joint / rows[:, order, None]
    all_change[measured] = 1 - np.trace(joint, axis1=1, axis2=2)
    return TransferResult(
        joint_probabilities=all_joint,
        transition_matrices=all_transition,
        change_probability=all_change,
        populations=populations[order],
        centers=centers[order] * scale + mean,
        sigma=float(np.sqrt(variance) * scale),
        singular_values=singular_values[: n_states + 1] / singular_values[0],
    )


def _normalized_moments(
    sums: NDArray[np.float64],
) -> tuple[NDArray[np.float64], complex, float]:
    """Normalize both readouts using shot-weighted pooled herald statistics."""
    counts = sums[:, 0, 0]
    pooled = sums.sum(axis=0)[:, 0] / counts.sum()
    mean = complex(pooled[1], pooled[2])
    variance = float(pooled[3] + pooled[5] - abs(mean) ** 2)
    if variance <= 0:
        raise AnalysisFailed("Readout centers are not identifiable without herald variance")
    scale = float(np.sqrt(variance))
    transform = _coordinate_transform(mean, scale)
    return transform @ (sums / counts[:, None, None]) @ transform.T, mean, scale


def _coordinate_transform(mean: complex, scale: float) -> NDArray[np.float64]:
    """Map raw quadratic features to pooled-herald centered, unitless features."""
    x, y = mean.real, mean.imag
    return np.array(
        [[1, 0, 0, 0, 0, 0],
         [-x, 1, 0, 0, 0, 0],
         [-y, 0, 1, 0, 0, 0],
         [x*x, -2*x, 0, 1, 0, 0],
         [x*y, -y, -x, 0, 1, 0],
         [y*y, 0, -2*y, 0, 0, 1]], dtype=np.float64
    ) / np.array([1, scale, scale, scale**2, scale**2, scale**2])[:, None]


def _lift(centers: NDArray[np.complex128], variance: float) -> NDArray[np.float64]:
    """Conditional quadratic feature means for isotropic Gaussian readout."""
    x, y = centers.real, centers.imag
    return np.column_stack(
        [np.ones(centers.size), x, y, x*x + variance, x*y, y*y + variance]
    )


def _fit_centers(
    projector: NDArray[np.float64], n_states: int
) -> tuple[NDArray[np.complex128], float]:
    """Find distinct lifted centers in the shared readout subspace."""
    axis = np.linspace(-4, 4, 101)
    grid = (axis[:, None] + 1j * axis[None, :]).ravel()
    best_score = np.inf
    initial = np.empty(2 * n_states + 1)
    for variance in np.linspace(0, 0.5, 31):
        lifted = _lift(grid, float(variance))
        distances = np.linalg.norm(lifted @ projector, axis=1) / np.linalg.norm(lifted, axis=1)
        candidates: list[complex] = []
        for index in np.argsort(distances):
            center = complex(grid[index])
            if all(abs(center - other) > 0.4 for other in candidates):
                candidates.append(center)
            if len(candidates) == n_states:
                break
        centers = np.array(candidates, dtype=np.complex128)
        score = float(np.sum((_lift(centers, float(variance)) @ projector)**2))
        if score < best_score:
            best_score = score
            initial = np.r_[np.column_stack([centers.real, centers.imag]).ravel(), variance]

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        centers = parameters[:-1:2] + 1j * parameters[1:-1:2]
        return (_lift(centers, float(parameters[-1])) @ projector).ravel()

    lower = np.r_[np.full(2 * n_states, -np.inf), 0.0]
    upper = np.r_[np.full(2 * n_states, np.inf), 0.5]
    fit = least_squares(residual, initial, bounds=(lower, upper), gtol=1e-12, ftol=1e-12, xtol=1e-12)
    centers = fit.x[:-1:2] + 1j * fit.x[1:-1:2]
    variance = float(fit.x[-1])
    lifted = _lift(centers, variance)
    relative_error = np.linalg.norm(lifted @ projector, axis=1) / np.linalg.norm(lifted, axis=1)
    if not fit.success or np.max(relative_error) > 0.05:
        raise AnalysisFailed("Readout identification lift residual did not converge")
    _require_distinct_centers(centers)
    return centers, variance


def _require_distinct_centers(centers: NDArray[np.complex128]) -> None:
    """Reject coincident centers and the unidentifiable collinear three-state case."""
    separations = np.abs(centers[:, None] - centers[None, :])
    distances = separations[np.triu_indices(len(centers), k=1)]
    if np.min(distances) <= 1e-3 * max(1.0, float(np.max(distances))):
        raise AnalysisFailed("Could not identify distinct readout centers")
    if len(centers) == 3:
        first, second = centers[1:] - centers[0]
        area = abs((first.conjugate() * second).imag)
        if area <= 1e-3 * abs(first) * abs(second):
            raise AnalysisFailed("Three collinear readout centers are not identifiable")
