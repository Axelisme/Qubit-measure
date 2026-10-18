"""Shared finite Hilbert-Fourier eigensystem for shaped and constant tones."""

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def eigensystem(energies, coupling, frequency, modes):
    """Diagonalize Sambe blocks and return the contiguous central-input rows.

    energies is the bare GHz diagonal, coupling is the positive-harmonic block,
    frequency is signed GHz and modes is the symmetric harmonic cutoff.
    Endpoint reconstruction and the energy-origin choice belong to callers.
    """
    size = len(energies)
    count = 2 * modes + 1
    matrix = np.zeros((count * size, count * size), dtype=np.complex128)
    for k in range(count):
        for j in range(size):
            matrix[k * size + j, k * size + j] = energies[j] + (k - modes) * frequency
        if k:
            matrix[k * size : (k + 1) * size, (k - 1) * size : k * size] = coupling
            matrix[(k - 1) * size : k * size, k * size : (k + 1) * size] = (
                coupling.conj().T
            )
    values, vectors = np.linalg.eigh(matrix)
    central = np.ascontiguousarray(vectors[modes * size : (modes + 1) * size])
    return values, vectors, central
