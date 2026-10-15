"""Exact jump algebra reuse, without time or Hamiltonian/bath splitting."""

import numpy as np

from .state import ComplexMatrix


class LindbladAction:
    """Internal solver algebra for one fixed positive matrix dimension.

    Static and arbitrary time-dependent baths both use this action. Every dynamic
    result is rechecked; even a callback reusing mutated storage remains valid.
    The optional dense gain workspace is at most 16 MiB. Larger models retain
    batched exact matrix multiplication instead of allocating n**4 storage.
    Instances initially have no jumps. They own no density or time history.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._operators = np.empty((0, dimension, dimension), dtype=np.complex128)
        self._adjoints = self._operators.copy()
        self._loss = np.zeros((dimension, dimension), dtype=np.complex128)
        self._gain: ComplexMatrix | None = None

    def replace(self, operators: tuple[ComplexMatrix, ...]) -> None:
        """Own finite square jumps; invalid shapes/values raise ValueError.

        Reuses algebra only if every entry equals the previously owned contents.
        All jumps must match the constructor dimension. Input arrays are never
        modified or retained by reference. Empty means no dissipative action.
        """
        n = self._dimension
        values = (
            np.asarray(operators, dtype=np.complex128)
            if operators
            else np.empty((0, n, n), dtype=np.complex128)
        )
        if values.shape[1:] != (n, n) or values.ndim != 3:
            raise ValueError("dynamic collapse shape must match basis")
        if not np.isfinite(values).all():
            raise ValueError("collapse operators must be finite")
        if np.array_equal(values, self._operators):
            return
        self._operators = values.copy()
        self._adjoints = values.conj().swapaxes(1, 2)
        self._loss = np.sum(self._adjoints @ values, axis=0)
        # C-order (i,j),(k,l): sum_a L_a[i,k] * conjugate(L_a[j,l]).
        self._gain = (
            np.einsum("aik,ajl->ijkl", values, values.conj(), optimize=True).reshape(
                n * n, n * n
            )
            if n <= 32 and len(values)
            else None
        )

    def apply(self, density: ComplexMatrix) -> ComplexMatrix:
        """Return sum L*rho*L†-{L†L,rho}/2 without modifying density.

        density uses the installed jumps' coordinates and constructor dimension.
        Shape mismatch raises ValueError. No repair, stepping or Hamiltonian
        evolution occurs; the caller adds this exact action to its generator.
        """
        if self._gain is not None:
            gain = (self._gain @ density.ravel()).reshape(density.shape)
        else:
            gain = np.sum(self._operators @ density @ self._adjoints, axis=0)
        return gain - (self._loss @ density + density @ self._loss) / 2
