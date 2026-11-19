"""Array ownership, density validation and readout density correction."""

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

Scalar = TypeVar("Scalar", bound=np.generic)

_HERMITIAN_ATOL = 1e-10
_DENSITY_ATOL = 1e-8


def owned(value: NDArray[Scalar], dtype: type[Scalar], name: str) -> NDArray[Scalar]:
    """Copy finite numeric data into immutable, explicitly typed storage."""
    array = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def integers(value: NDArray[np.int64], name: str) -> NDArray[np.int64]:
    """Own integer indices without silently truncating floating inputs."""
    if np.asarray(value).dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    return owned(value, np.int64, name)


def shape(value: NDArray[Scalar], expected: tuple[int, ...], name: str) -> None:
    """Reject an array with a different shape."""
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def hermitian(value: NDArray[np.complex128], name: str) -> NDArray[np.complex128]:
    """Own a nonempty square Hermitian matrix; absolute tolerance 1e-10."""
    matrix = owned(value, np.complex128, name)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.allclose(matrix, matrix.conj().T, rtol=0, atol=_HERMITIAN_ATOL):
        raise ValueError(f"{name} must be Hermitian")
    return matrix


def density(value: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Own a Hermitian, positive, trace-one density within tolerance 1e-8."""
    matrix = hermitian(value, "initial density")
    if (
        abs(np.trace(matrix) - 1) > _DENSITY_ATOL
        or np.linalg.eigvalsh(matrix).min() < -_DENSITY_ATOL
    ):
        raise ValueError("initial density must be positive and trace one")
    return matrix


def project_readout_densities(
    matrices: NDArray[np.complex128], points: NDArray[np.int64], budget: float
) -> None:
    """Repair invalid readout densities before publishing an evolve result.

    matrices is writable (n,M,M) in the DC eigenbasis; points holds the n
    aligned caller IDs. Valid matrices are left unchanged. For invalid
    matrices, clip negative eigenvalues of the Hermitian part and normalize
    the positive weights. Raise RuntimeError with the point ID if no positive
    weight can be normalized or the maximum absolute complex entry change
    exceeds the call's budget. Nonfinite input raises ValueError.
    """
    if not np.isfinite(matrices).all():
        raise ValueError("result densities must be finite")
    adjoint = matrices.swapaxes(-1, -2).conj()
    invalid = (
        (np.max(np.abs(matrices - adjoint), axis=(1, 2)) > _HERMITIAN_ATOL)
        | (np.abs(np.trace(matrices, axis1=1, axis2=2) - 1) > _DENSITY_ATOL)
        | (np.linalg.eigvalsh(matrices)[:, 0] < -_DENSITY_ATOL)
    )
    for row in np.flatnonzero(invalid):
        values, vectors = np.linalg.eigh((matrices[row] + adjoint[row]) / 2)
        weights = np.maximum(values, 0)
        trace = weights.sum()
        if not np.isfinite(trace) or trace <= 0:
            raise RuntimeError(f"point {points[row]}: cannot normalize readout density")
        projected = (vectors * (weights / trace)) @ vectors.conj().T
        correction = float(np.max(np.abs(projected - matrices[row])))
        if correction > budget:
            raise RuntimeError(
                f"point {points[row]}: readout density correction {correction:.6g} "
                f"exceeds budget {budget:.6g}"
            )
        matrices[row] = projected


def ragged(offsets: NDArray[np.int64], times: NDArray[np.float64]) -> None:
    """Check complete, nonempty per-point spans and increasing local times."""
    if offsets.ndim != 1 or not len(offsets) or offsets[0] != 0:
        raise ValueError("offsets must be a vector beginning at zero")
    if times.ndim != 1 or offsets[-1] != len(times) or np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets must increase and cover all samples")
    if np.any(times < 0):
        raise ValueError("sample times must be nonnegative")
    # Times may reset at a point boundary, but not inside a point.
    differences = np.diff(times).copy()
    differences[offsets[1:-1] - 1] = 1
    if np.any(differences <= 0):
        raise ValueError("sample times must increase within each point")
