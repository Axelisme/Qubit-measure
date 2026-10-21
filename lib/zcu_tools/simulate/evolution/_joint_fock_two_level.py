"""Research protocol: analytic one-jump prefill and shared total-N probe rows."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.linalg import expm
from scipy.stats import poisson

from ._joint_fock_numeric import (
    Basis,
    ComplexArray,
    RealArray,
    cavity_generator,
    cavity_operators,
    coherent,
    commutator,
    intrinsic_bath,
    rk4,
)

if TYPE_CHECKING:
    from .joint_fock import JointFockRequest


@dataclass(frozen=True)
class PointerFields:
    """Conditional field generators, steady pointers and one fixed displacement."""

    matrices: ComplexArray
    steady: ComplexArray
    displacement: ComplexArray


def pointer_fields(
    request: "JointFockRequest", rf: float, *, frequencies: RealArray | None = None
) -> PointerFields:
    """Build the physical g/e no-flip fields at one rf, in sqrt(photons)."""
    v = np.sqrt(request.kappa_external_per_ns / 2)
    damping = np.outer(v, v) + np.diag(request.kappa_internal_per_ns / 2)
    matrices = np.array(
        [
            -damping - 2j * np.pi * np.diag(f - rf)
            for f in (request.frequencies_ghz if frequencies is None else frequencies)
        ]
    )
    steady = np.zeros((2, 2), complex)
    if request.incident_per_sqrt_ns != 0:
        try:
            steady = np.array(
                [
                    np.linalg.solve(-m, v * request.incident_per_sqrt_ns)
                    for m in matrices
                ]
            )
        except np.linalg.LinAlgError as error:
            raise ValueError("Joint-Fock steady pointer is singular") from error
    return PointerFields(matrices, steady, np.mean(steady, axis=0))


def pointer_cutoff(
    request: "JointFockRequest", fields: PointerFields, *, include_vacuum: bool = False
) -> tuple[int, RealArray]:
    """Sample no-flip/one-jump coherent means and return tail<=1e-3 N, plus one.

    This heuristic is not a density truncation error bound. Analytic two-level
    prefill excludes earlier physical vacuum, which is never propagated there.
    """
    start = 0.0 if include_vacuum else request.prefill_duration_ns
    times = np.linspace(
        start, request.prefill_duration_ns + request.probe_duration_ns, 129
    )
    means = [
        max(
            float(np.linalg.norm(a - expm(m * t) @ a - fields.displacement) ** 2)
            for t in times
        )
        for m, a in zip(fields.matrices, fields.steady, strict=True)
    ]
    jumped_mean = 0.0
    for time in np.linspace(0, request.prefill_duration_ns, 65):
        at = fields.steady[1] - expm(fields.matrices[1] * time) @ fields.steady[1]
        for after in np.linspace(
            request.prefill_duration_ns - time,
            request.prefill_duration_ns + request.probe_duration_ns - time,
            33,
        ):
            jumped = fields.steady[0] + expm(fields.matrices[0] * after) @ (
                at - fields.steady[0]
            )
            jumped_mean = max(
                jumped_mean, float(np.linalg.norm(jumped - fields.displacement) ** 2)
            )
    means.append(jumped_mean)
    for order in range(20):
        if poisson.sf(order, max(means)) <= 1e-3:
            return order + 1, np.array(means)
    raise ValueError("Sampled pointer tail requires unsupported cutoff N > 20")


def nominal_pi(request: "JointFockRequest") -> ComplexArray:
    """Finite resonant pulse from ground, retaining direct rates and coherence."""
    down, up, gamma = request.down_per_ns, request.up_per_ns, request.gamma2_per_ns
    omega = request.pi_rabi_rad_per_ns
    matrix = np.array(
        [
            [-up, down, 0, -omega],
            [up, -down, 0, omega],
            [0, 0, -gamma, 0],
            [omega / 2, -omega / 2, 0, -gamma],
        ]
    )
    pg, pe, x, y = expm(matrix * request.pi_duration_ns) @ [1.0, 0, 0, 0]
    return np.array([[pg, x + 1j * y], [x - 1j * y, pe]])


def analytic_prefill(
    request: "JointFockRequest", fields: PointerFields, basis: Basis
) -> ComplexArray:
    """Two F-order density columns with the reference's one-jump/qubit phase."""
    size, duration = len(basis), request.prefill_duration_ns
    down = request.down_per_ns
    field = [
        a - expm(m * duration) @ a
        for m, a in zip(fields.matrices, fields.steady, strict=True)
    ]
    vectors = [coherent(a - fields.displacement, basis) for a in field]
    vacuum = [np.outer(a, a.conj()) for a in vectors]
    nodes, weights = leggauss(64)
    integral, jump_density = 0j, np.zeros((size, size), complex)
    delta = 2 * np.pi * (request.frequencies_ghz[1] - request.frequencies_ghz[0])
    for time, weight in zip(
        duration * (nodes + 1) / 2, duration * weights / 2, strict=True
    ):
        at = [
            a - expm(m * time) @ a
            for m, a in zip(fields.matrices, fields.steady, strict=True)
        ]
        jumped = fields.steady[0] + expm(fields.matrices[0] * (duration - time)) @ (
            at[1] - fields.steady[0]
        )
        vector = coherent(jumped - fields.displacement, basis)
        jump_density += (
            down * weight * np.exp(-down * time) * np.outer(vector, vector.conj())
        )
        integral += weight * np.sum(at[1].conj() * delta * at[0])
    overlap = np.exp(
        -0.5
        * (
            np.vdot(field[0] - fields.displacement, field[0] - fields.displacement)
            + np.vdot(field[1] - fields.displacement, field[1] - fields.displacement)
        )
        + np.vdot(field[1] - fields.displacement, field[0] - fields.displacement)
    )
    factor = np.exp(1j * integral - request.gamma2_per_ns * duration) / overlap
    rotation = np.repeat(
        [
            np.exp(
                1j
                * np.pi
                * request.qubit_frequency_ghz
                * (request.pi_duration_ns + duration)
            ),
            np.exp(
                -1j
                * np.pi
                * request.qubit_frequency_ghz
                * (request.pi_duration_ns + duration)
            ),
        ],
        size,
    )
    states = []
    for qubit in (np.diag([1.0, 0.0]), nominal_pi(request)):
        density = np.zeros((2 * size, 2 * size), complex)
        density[:size, :size] = qubit[0, 0] * vacuum[0] + qubit[1, 1] * jump_density
        density[size:, size:] = qubit[1, 1] * np.exp(-down * duration) * vacuum[1]
        density[:size, size:] = (
            qubit[0, 1] * factor * np.outer(vectors[0], vectors[1].conj())
        )
        density[size:, :size] = density[:size, size:].conj().T
        density *= rotation[:, None] * rotation.conj()[None, :]
        states.append(density.ravel(order="F"))
    result = np.column_stack(states)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite analytic joint-Fock prefill")
    return result


@dataclass(frozen=True)
class FockRow:
    """One rf's final densities (preparation,qf,d,d) and cutoff inventory."""

    densities: ComplexArray
    basis: Basis
    sampled_means: RealArray
    initial_trace_error: RealArray
    initial_edge_population: RealArray
    vacuum_trace_error: float
    vacuum_edge_population: float


def two_level_row(request: "JointFockRequest", rf: float) -> FockRow:
    """Prepare each rf once; propagate all probes from the same two columns."""
    fields = pointer_fields(request, rf)
    order, means = pointer_cutoff(request, fields)
    basis = tuple((n, m) for n in range(order + 1) for m in range(order + 1 - n))
    size = len(basis)
    operators = cavity_operators(basis, fields.displacement)
    rates = np.array([[0, request.down_per_ns], [request.up_per_ns, 0]])
    pure = np.array(
        [
            [0, request.gamma2_per_ns - (request.down_per_ns + request.up_per_ns) / 2],
            [request.gamma2_per_ns - (request.down_per_ns + request.up_per_ns) / 2, 0],
        ]
    )
    base = cavity_generator(
        request.frequencies_ghz,
        rf,
        operators,
        request.kappa_external_per_ns,
        request.kappa_internal_per_ns,
        request.incident_per_sqrt_ns,
    ) + intrinsic_bath(rates, pure, size)
    identity = sparse.csr_matrix(np.eye(size))
    pe = sparse.csr_matrix(sparse.kron(sparse.diags([0.0, 1.0]), identity))
    sx = sparse.csr_matrix(sparse.kron(np.array([[0, 1], [1, 0]]), identity))
    detuning, transverse = commutator(pe), commutator(sx / 2)
    states = analytic_prefill(request, fields, basis)
    initial = np.stack(
        [states[:, b].reshape(2 * size, 2 * size, order="F") for b in (0, 1)]
    )
    edge = np.tile([n + m == order for n, m in basis], 2)
    outputs = []
    for qf in request.qubit_carriers_ghz:
        generator = sparse.csr_matrix(
            base
            + 2 * np.pi * (request.qubit_frequency_ghz - qf) * detuning
            + request.probe_rabi_rad_per_ns * transverse
        )
        output = rk4(generator, states, request.probe_duration_ns)
        outputs.append(
            np.stack(
                [output[:, b].reshape(2 * size, 2 * size, order="F") for b in (0, 1)]
            )
        )
    vacuum = coherent(-fields.displacement, basis)
    return FockRow(
        np.stack(outputs, axis=1),
        basis,
        means,
        abs(np.trace(initial, axis1=-2, axis2=-1) - 1),
        initial.diagonal(axis1=-2, axis2=-1).real[:, edge].sum(axis=-1),
        float(abs(np.vdot(vacuum, vacuum) - 1)),
        float(np.sum(abs(vacuum[np.array([n + m == order for n, m in basis])]) ** 2)),
    )
