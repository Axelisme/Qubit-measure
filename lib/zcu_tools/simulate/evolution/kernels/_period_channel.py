"""Complete secular period channels with sequential and binary-power stepping."""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(nogil=True, cache=True)
def _dissipate_into(rho, population, coherence, result, diagonal, updated):
    for i in range(len(rho)):
        diagonal[i] = rho[i, i]
        for j in range(len(rho)):
            result[i, j] = rho[i, j] * coherence[i, j]
    np.dot(population, diagonal, out=updated)
    for i in range(len(rho)):
        result[i, i] = updated[i]


@njit(nogil=True, cache=True)
def _advance_loop(rho, unitary, population, coherence, count):
    # Preserve each GEMM and half-bath rounding while reusing worker-local buffers.
    adjoint = unitary.conj().T.copy()
    state = rho.copy()
    first = np.empty(rho.shape, dtype=np.complex128)
    second = np.empty(rho.shape, dtype=np.complex128)
    diagonal = np.empty(len(rho), dtype=np.complex128)
    updated = np.empty(len(rho), dtype=np.complex128)
    for _ in range(count):
        _dissipate_into(state, population, coherence, first, diagonal, updated)
        np.dot(unitary, first, out=second)
        np.dot(second, adjoint, out=state)
        _dissipate_into(state, population, coherence, first, diagonal, updated)
        state, first = first, state
    return state


@njit(nogil=True, cache=True)
def _hermitian_coordinates(size):
    # Diagonal entries, then real/imaginary off-diagonal basis pairs.
    a = np.empty(size * size, dtype=np.int64)
    b = np.empty(size * size, dtype=np.int64)
    kind = np.zeros(size * size, dtype=np.int64)
    for i in range(size):
        a[i] = b[i] = i
    index = size
    for i in range(size):
        for j in range(i + 1, size):
            a[index] = a[index + 1] = i
            b[index] = b[index + 1] = j
            kind[index] = 1
            kind[index + 1] = 2
            index += 2
    return a, b, kind


@njit(nogil=True, cache=True)
def _real_channel(unitary, population, coherence, coordinates):
    # U(.)U† in a Hermitian basis is real. Apply both secular half maps to
    # its input/output coordinates, without commuting dissipation through U.
    a, b, kind = coordinates
    size = len(unitary)
    matrix = np.empty((size * size, size * size), dtype=np.float64)
    for r in range(len(a)):
        for c in range(len(a)):
            z = unitary[a[r], a[c]] * unitary[b[r], b[c]].conjugate()
            if kind[c] == 1:
                z += unitary[a[r], b[c]] * unitary[b[r], a[c]].conjugate()
            elif kind[c] == 2:
                z = 1j * (z - unitary[a[r], b[c]] * unitary[b[r], a[c]].conjugate())
            matrix[r, c] = z.imag if kind[r] == 2 else z.real
    p = population.real.copy()
    matrix[:size, :] = p @ matrix[:size, :].copy()
    matrix[:, :size] = matrix[:, :size].copy() @ p
    for r in range(size, len(a)):
        for c in range(len(a)):
            matrix[r, c] *= coherence[a[r], b[r]]
    for c in range(size, len(a)):
        for r in range(len(a)):
            matrix[r, c] *= coherence[a[c], b[c]]
    return matrix


@njit(nogil=True, cache=True)
def _advance_power(rho, unitary, population, coherence, count):
    coordinates = _hermitian_coordinates(len(rho))
    a, b, kind = coordinates
    matrix = _real_channel(unitary, population, coherence, coordinates)
    state = np.empty(len(a), dtype=np.float64)
    residual = np.empty(len(a), dtype=np.float64)
    for i in range(len(a)):
        if kind[i] == 0:
            z = rho[a[i], b[i]]
        elif kind[i] == 1:
            z = (rho[a[i], b[i]] + rho[b[i], a[i]]) / 2
        else:
            z = (rho[a[i], b[i]] - rho[b[i], a[i]]) / (2j)
        # Complex coordinates span arbitrary matrices, not just Hermitian
        # density. Keep their imaginary part as a second real right-hand side.
        state[i] = z.real
        residual[i] = z.imag
    while count:
        if count & 1:
            state = matrix @ state
            residual = matrix @ residual
        count //= 2
        if count:
            matrix = matrix @ matrix
    result = np.empty_like(rho)
    for i in range(len(a)):
        z = state[i] + 1j * residual[i]
        if kind[i] == 0:
            result[a[i], b[i]] = z
        elif kind[i] == 1:
            w = state[i + 1] + 1j * residual[i + 1]
            result[a[i], b[i]] = z + 1j * w
            result[b[i], a[i]] = z - 1j * w
    return result


@njit(nogil=True, cache=True)
def advance(
    rho: NDArray[np.complex128],
    unitary: NDArray[np.complex128],
    population: NDArray[np.complex128],
    coherence: NDArray[np.float64],
    count: int,
) -> NDArray[np.complex128]:
    """Apply count complete secular D/2, UρU†, D/2 periods.

    rho, unitary and the population/coherence half maps share shape (d, d).
    The caller supplies valid numeric inputs and a nonnegative integer count.
    rho may be arbitrary complex, not necessarily Hermitian or trace one.
    Inputs are not modified; count zero returns rho itself, otherwise a new
    array is returned. NumPy and Numba errors propagate without translation.
    """
    if count == 0:
        return rho
    if len(rho) == 16 and count >= 4096:
        return _advance_power(rho, unitary, population, coherence, count)
    return _advance_loop(rho, unitary, population, coherence, count)


@njit(nogil=True, cache=True)
def advance_segments(
    rho: NDArray[np.complex128],
    unitaries: NDArray[np.complex128],
    population: NDArray[np.complex128],
    coherence: NDArray[np.float64],
    count: int,
) -> NDArray[np.complex128]:
    """Repeat an ordered sequence of secular D/2, UρU†, D/2 segments.

    unitaries has shape (m, d, d); rho and the shared population/coherence
    half maps have shape (d, d). count is a nonnegative number of complete
    periods, each traversing all m segments. The caller supplies valid numeric
    inputs. rho may be arbitrary complex, not necessarily Hermitian or trace
    one. Inputs are not modified; zero count or an empty sequence returns rho
    itself, otherwise a new array is returned. NumPy and Numba errors propagate
    without translation.
    """
    for _ in range(count):
        for unitary in unitaries:
            rho = advance(rho, unitary, population, coherence, 1)
    return rho
