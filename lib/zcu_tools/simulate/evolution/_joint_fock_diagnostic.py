"""Small multilevel joint-Fock diagnostic with full-parent virtual coefficients."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.stats import poisson

from ._joint_fock_numeric import (
    ComplexArray,
    RealArray,
    cavity_generator,
    cavity_operators,
    coherent,
    intrinsic_bath,
    rk4,
)
from ._joint_fock_two_level import FockRow, pointer_cutoff, pointer_fields
from .kernels.multitone_rwa import (
    RWACoefficients,
    interaction_hamiltonian,
    rwa_coefficients,
)

if TYPE_CHECKING:
    from .joint_fock import JointFockDiagnostic, JointFockRequest


@dataclass(frozen=True)
class DrivenSystem:
    """Projected interaction Hamiltonian plus the constant joint Lindblad part."""

    energies: RealArray
    coefficients: RWACoefficients
    frequency: float
    gain: float
    cavity_size: int
    constant: sparse.csr_matrix

    def rhs(self, time: float, states: ComplexArray) -> ComplexArray:
        """Derivative in local ns and the system's interaction picture."""
        levels, size = len(self.energies), self.cavity_size
        dimension = levels * size
        density = states.reshape(dimension, dimension, -1, order="F").reshape(
            levels, size, levels, size, -1
        )
        h = interaction_hamiltonian(
            time,
            self.energies,
            self.coefficients,
            np.array([self.gain * np.exp(2j * np.pi * self.frequency * time)]),
        )
        left = np.einsum("ik,kajbp->iajbp", h, density)
        right = np.einsum("iakbp,kj->iajbp", density, h)
        driven = (
            (-2j * np.pi * (left - right))
            .reshape(dimension, dimension, -1)
            .reshape(dimension * dimension, -1, order="F")
        )
        return np.asarray(self.constant @ states + driven, dtype=np.complex128)


def _coefficients(
    diagnostic: "JointFockDiagnostic", frequency: float
) -> RWACoefficients:
    # Virtual sums are formed over every full-model state BEFORE projection.
    full = rwa_coefficients(diagnostic.model, (diagnostic.drive_port,), (frequency,))
    indices = list(diagnostic.retained_indices)
    return RWACoefficients(
        full.positive[:, indices][:, :, indices].copy(),
        full.negative[:, indices][:, :, indices].copy(),
        full.diagonal[:, indices].copy(),
    )


def _driven_rk4(
    system: DrivenSystem, initial: ComplexArray, duration: float, step_ns: float
) -> ComplexArray:
    states = initial.copy()
    steps = max(1, int(np.ceil(duration / step_ns)))
    dt = duration / steps
    for step in range(steps):
        time = step * dt
        k1 = system.rhs(time, states)
        k2 = system.rhs(time + dt / 2, states + dt / 2 * k1)
        k3 = system.rhs(time + dt / 2, states + dt / 2 * k2)
        k4 = system.rhs(time + dt, states + dt * k3)
        states += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if not np.isfinite(states).all():
        raise FloatingPointError("Nonfinite diagnostic joint-Fock RK4 density")
    return _restore_endpoint(states, system.energies, system.cavity_size, duration)


def _restore_endpoint(
    states: ComplexArray, energies: RealArray, cavity_size: int, duration: float
) -> ComplexArray:
    phase = np.repeat(np.exp(-2j * np.pi * energies * duration), cavity_size)
    return states * (phase[:, None] * phase.conj()[None, :]).ravel(order="F")[:, None]


def diagnostic_row(
    request: "JointFockRequest", rf: float, diagnostic: "JointFockDiagnostic"
) -> FockRow:
    """Physical vacuum -> numeric pi -> numeric prefill -> independent probes."""
    indices = list(diagnostic.retained_indices)
    energies = diagnostic.model.energies_ghz[indices]
    rates = diagnostic.model.rates_per_ns[np.ix_(indices, indices)]
    pure = diagnostic.model.dephasing_per_ns[np.ix_(indices, indices)]
    fields = pointer_fields(request, rf, frequencies=request.frequencies_ghz[[0, 2]])
    order, means = pointer_cutoff(request, fields, include_vacuum=True)
    basis = tuple((n, m) for n in range(order + 1) for m in range(order + 1 - n))
    size, levels = len(basis), len(indices)
    vector = coherent(-fields.displacement, basis)
    if (
        poisson.sf(order, float(np.vdot(fields.displacement, fields.displacement).real))
        > 1e-3
    ):
        raise ValueError(
            "Initial physical vacuum cannot be represented at selected cutoff"
        )
    vacuum = np.outer(vector, vector.conj())
    ground = np.zeros((levels, levels), complex)
    ground[0, 0] = 1
    pi_system = DrivenSystem(
        energies,
        _coefficients(diagnostic, request.qubit_frequency_ghz),
        request.qubit_frequency_ghz,
        diagnostic.pi_gain,
        1,
        intrinsic_bath(rates, pure, 1),
    )
    pi = _driven_rk4(
        pi_system,
        ground.ravel(order="F")[:, None],
        request.pi_duration_ns,
        diagnostic.step_ns,
    )[:, 0].reshape(levels, levels, order="F")
    states = np.asarray(
        np.column_stack(
            [np.kron(rho, vacuum).ravel(order="F") for rho in (ground, pi)]
        ),
        dtype=np.complex128,
    )
    operators = cavity_operators(basis, fields.displacement)
    base = cavity_generator(
        request.frequencies_ghz[indices],
        rf,
        operators,
        request.kappa_external_per_ns,
        request.kappa_internal_per_ns,
        request.incident_per_sqrt_ns,
    ) + intrinsic_bath(rates, pure, size)
    states = rk4(
        sparse.csr_matrix(base), states, request.prefill_duration_ns, diagnostic.step_ns
    )
    states = _restore_endpoint(states, energies, size, request.prefill_duration_ns)
    dimension = levels * size
    initial = np.stack(
        [states[:, b].reshape(dimension, dimension, order="F") for b in (0, 1)]
    )
    outputs = []
    for qf in request.qubit_carriers_ghz:
        system = DrivenSystem(
            energies,
            _coefficients(diagnostic, float(qf)),
            float(qf),
            diagnostic.probe_gain,
            size,
            sparse.csr_matrix(base),
        )
        output = _driven_rk4(
            system, states, request.probe_duration_ns, diagnostic.step_ns
        )
        outputs.append(
            np.stack(
                [output[:, b].reshape(dimension, dimension, order="F") for b in (0, 1)]
            )
        )
    edge = np.array([n + m == order for n, m in basis])
    return FockRow(
        np.stack(outputs, axis=1),
        basis,
        means,
        abs(np.trace(initial, axis1=-2, axis2=-1) - 1),
        initial.diagonal(axis1=-2, axis2=-1)
        .real[:, np.tile(edge, levels)]
        .sum(axis=-1),
        float(abs(np.trace(vacuum) - 1)),
        float(vacuum.diagonal().real[edge].sum()),
    )
