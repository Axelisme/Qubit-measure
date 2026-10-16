"""Array ownership and structural validation shared by engine records."""

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

Scalar = TypeVar("Scalar", bound=np.generic)


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
    if not np.allclose(matrix, matrix.conj().T, rtol=0, atol=1e-10):
        raise ValueError(f"{name} must be Hermitian")
    return matrix


def density(value: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Own a Hermitian, positive, trace-one density within tolerance 1e-8."""
    matrix = hermitian(value, "initial density")
    if abs(np.trace(matrix) - 1) > 1e-8 or np.linalg.eigvalsh(matrix).min() < -1e-8:
        raise ValueError("initial density must be positive and trace one")
    return matrix


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
