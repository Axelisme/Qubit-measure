"""Explicit fixed-coordinate density states and standalone population conversion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]


def owned_matrix(value: ComplexMatrix, name: str) -> ComplexMatrix:
    """Copy a finite complex square matrix into read-only storage."""
    result = np.array(value, dtype=np.complex128, copy=True)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Basis:
    """Ordered orthonormal basis in a common ambient coordinate space.

    identity: nonempty caller-owned reproducible basis identifier, not a cache key.
    labels: unique state names in column order of vectors.
    vectors: dimensionless square complex matrix; columns are basis vectors in
        the common ambient coordinates shared by source and target bases.
        Construction owns a read-only copy. Invalid/nonunitary input raises
        ValueError; tolerance is 1e-10, not a basis-convergence certification.
    """

    identity: str
    labels: tuple[str, ...]
    vectors: ComplexMatrix

    def __post_init__(self) -> None:
        vectors = owned_matrix(self.vectors, "basis vectors")
        n = vectors.shape[0]
        if not self.identity or len(self.labels) != n or len(set(self.labels)) != n:
            raise ValueError("basis identity and unique labels must match dimension")
        if not np.allclose(vectors.conj().T @ vectors, np.eye(n), rtol=0, atol=1e-10):
            raise ValueError("basis vectors must be orthonormal")
        object.__setattr__(self, "vectors", vectors)


@dataclass(frozen=True)
class DensityState:
    """Density matrix explicitly attached to its representation basis.

    basis: row/column order and common ambient coordinates of density.
    density: dimensionless (state,state) complex128 matrix in basis, copied and
        made read-only. Must be finite, Hermitian, trace one and positive within
        1e-8 numerical tolerance. Invalid input raises ValueError. Values are
        never clipped, symmetrized or renormalized, including solver outputs.
    """

    basis: Basis
    density: ComplexMatrix

    def __post_init__(self) -> None:
        density = owned_matrix(self.density, "density")
        if density.shape != self.basis.vectors.shape:
            raise ValueError("density shape must match basis")
        if not np.allclose(density, density.conj().T, rtol=0, atol=1e-8):
            raise ValueError("density must be Hermitian")
        if abs(np.trace(density) - 1) > 1e-8:
            raise ValueError("density trace must be one")
        if np.linalg.eigvalsh(density).min() < -1e-8:
            raise ValueError("density must be positive semidefinite")
        object.__setattr__(self, "density", density)


@dataclass(frozen=True)
class Populations:
    """Standalone target-basis population vector, without ADC responsibilities.

    basis: target basis and state order used to compute values.
    values: dimensionless real (state,) vector, owned/read-only. Tiny negative
        numerical residuals are retained, not repaired. Consumers choose their
        required probability-validation tolerance; no implicit conversion to a
        different basis or state order occurs.
    """

    basis: Basis
    values: RealVector

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.shape != (len(self.basis.labels),) or not np.isfinite(values).all():
            raise ValueError("population shape/values must match basis")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


def populations(state: DensityState, target_basis: Basis) -> Populations:
    """Return diagonal of density after an explicit source-to-target transform.

    state.basis and target_basis must use the same ambient coordinate space and
    dimension. The caller owns that physical correspondence. Output order is
    target_basis.labels, with no projection, clipping or normalization. Inputs
    remain unchanged; mismatched dimensions raise ValueError. This operation
    does not model measurement trajectories or time evolution.
    """
    if state.basis.vectors.shape != target_basis.vectors.shape:
        raise ValueError("source and target basis dimensions must match")
    transform = target_basis.vectors.conj().T @ state.basis.vectors
    transformed = transform @ state.density @ transform.conj().T
    return Populations(target_basis, np.diag(transformed).real)
