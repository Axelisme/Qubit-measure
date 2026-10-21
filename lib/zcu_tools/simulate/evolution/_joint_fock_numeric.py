"""Total-excitation operators and fixed-step sparse propagation for joint Fock."""

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy import sparse

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Basis = tuple[tuple[int, int], ...]


def commutator(h: sparse.csr_matrix) -> sparse.csr_matrix:
    """Angular-frequency Hamiltonian to an F-order Liouvillian."""
    identity = sparse.csr_matrix(np.eye(len(h.indptr) - 1))
    return sparse.csr_matrix(
        -1j * (sparse.kron(identity, h) - sparse.kron(h.T, identity))
    )


def dissipator(op: sparse.csr_matrix) -> sparse.csr_matrix:
    """Unit-rate Lindblad dissipator in F-order density coordinates."""
    identity = sparse.csr_matrix(np.eye(len(op.indptr) - 1))
    square = op.conj().T @ op
    return sparse.csr_matrix(
        sparse.kron(op.conj(), op)
        - (sparse.kron(identity, square) + sparse.kron(square.T, identity)) / 2
    )


@njit(cache=True)
def _product(data, indices, indptr, states):
    result = np.zeros_like(states)
    for row in range(len(indptr) - 1):
        for offset in range(indptr[row], indptr[row + 1]):
            for branch in range(states.shape[1]):
                result[row, branch] += data[offset] * states[indices[offset], branch]
    return result


@njit(cache=True)
def _fixed_rk4(data, indices, indptr, states, duration):
    result = states.copy()
    steps = max(1, int(np.ceil(duration)))
    dt = duration / steps
    for _ in range(steps):
        k1 = _product(data, indices, indptr, result)
        k2 = _product(data, indices, indptr, result + dt / 2 * k1)
        k3 = _product(data, indices, indptr, result + dt / 2 * k2)
        k4 = _product(data, indices, indptr, result + dt * k3)
        result += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return result


def rk4(
    generator: sparse.csr_matrix, states: ComplexArray, duration: float
) -> ComplexArray:
    """Advance every independent column with fixed RK4 <=1 ns, without repair."""
    result = _fixed_rk4(
        generator.data,
        generator.indices,
        generator.indptr,
        np.ascontiguousarray(states),
        duration,
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite joint-Fock RK4 density")
    return result


def coherent(amplitude: ComplexArray, basis: Basis) -> ComplexArray:
    """Normalized finite total-N coherent vector; reject unrepresentable input."""
    order = max(n + m for n, m in basis)
    powers = np.ones((2, order + 1), dtype=np.complex128)
    for n in range(1, order + 1):
        powers[:, n] = powers[:, n - 1] * amplitude / np.sqrt(n)
    vector = np.array(
        [powers[0, n] * powers[1, m] for n, m in basis], dtype=np.complex128
    )
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Initial coherent state cannot be represented")
    return vector / norm


def cavity_operators(
    basis: Basis, displacement: ComplexArray
) -> tuple[sparse.csr_matrix, ...]:
    """Shifted cavity ladders formed inside the finite trace-preserving basis."""
    indices = {pair: index for index, pair in enumerate(basis)}
    identity = sparse.csr_matrix(np.eye(len(basis)))
    operators = []
    for mode in (0, 1):
        ladder = np.zeros((len(basis), len(basis)), complex)
        for column, pair in enumerate(basis):
            if pair[mode]:
                target = list(pair)
                target[mode] -= 1
                ladder[indices[(target[0], target[1])], column] = np.sqrt(pair[mode])
        operators.append(sparse.csr_matrix(ladder) + displacement[mode] * identity)
    return tuple(sparse.csr_matrix(op) for op in operators)


def cavity_generator(
    frequencies: RealArray,
    carrier: float,
    operators: tuple[sparse.csr_matrix, ...],
    external: RealArray,
    internal: RealArray,
    incident: float,
) -> sparse.csr_matrix:
    """Shared two-port line plus conditional cavities, without intrinsic baths."""
    system_identity = sparse.csr_matrix(np.eye(len(frequencies)))
    lifted = [sparse.csr_matrix(sparse.kron(system_identity, op)) for op in operators]
    v = np.sqrt(external / 2)
    port = sparse.csr_matrix(v[0] * lifted[0] + v[1] * lifted[1])
    h = sparse.csr_matrix(1j * incident * (port.conj().T - port))
    for mode, op in enumerate(operators):
        h += sparse.kron(
            sparse.diags(2 * np.pi * (frequencies[:, mode] - carrier)), op.conj().T @ op
        )
    generator = commutator(h) + 2 * dissipator(port)
    for rate, op in zip(internal, lifted, strict=True):
        generator += rate * dissipator(op)
    return sparse.csr_matrix(generator)


def intrinsic_bath(
    rates: RealArray, pure: RealArray, cavity_size: int
) -> sparse.csr_matrix:
    """Restrict rates before computing exits; no outside-level sink is invented."""
    exits = rates.sum(axis=0)
    decay = (exits[:, None] + exits[None, :]) / 2 + pure
    expanded = np.repeat(np.repeat(decay, cavity_size, axis=0), cavity_size, axis=1)
    result = sparse.csr_matrix(sparse.diags(-expanded.ravel(order="F"), dtype=complex))
    identity = sparse.csr_matrix(np.eye(cavity_size))
    for target, source in np.argwhere(rates > 0):
        op = sparse.csr_matrix(([1.0], ([target], [source])), shape=rates.shape)
        jump = sparse.kron(op, identity)
        result += rates[target, source] * sparse.kron(jump.conj(), jump)
    return sparse.csr_matrix(result)
