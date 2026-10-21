"""Explicit joint qubit/two-resonator prototype, separate from Engine and ADC."""

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import expm

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class JointFockRequest:
    """Numeric two-state spectrum in GHz, ns and radians, with two preparations.

    frequencies_ghz: (2,2) absolute conditional cavity frequencies, rows g/e.
    kappa_external_per_ns, kappa_internal_per_ns: (2,) energy decay rates.
        The external ports share reciprocal co-located two-port damping.
    resonator_carriers_ghz, qubit_carriers_ghz: nonempty 1-D scan coordinates.
    qubit_frequency_ghz: physical g/e gap (FTF dynamic indices [0,2]).
    incident_per_sqrt_ns: real coherent incident field, constant during prefill
        and probe. Each rf starts in physical vacuum, independently.
    down_per_ns, up_per_ns: direct e->g and g->e rates, not outside-level exit.
    gamma2_per_ns: full intrinsic coherence decay, at least (down+up)/2.
    pi_duration_ns, pi_rabi_rad_per_ns: finite resonant preparation pulse on
        ground before cavity illumination. The other preparation stays ground.
    prefill_duration_ns: illumination before the weak probe; analytic one-jump
        preparation requires up_per_ns * prefill_duration_ns <= 1e-12.
    probe_duration_ns, probe_rabi_rad_per_ns: constant weak qubit probe.
        Carriers reset locally at probe onset, as in the reference protocol.

    Arrays are copied read-only. Nonfinite/negative rates or durations, bad
    shapes, negative pure dephasing and nonnegligible upward prefill raise
    ValueError. There are no acquisition, ADC, noise or environment inputs.
    """

    frequencies_ghz: RealArray
    kappa_external_per_ns: RealArray
    kappa_internal_per_ns: RealArray
    resonator_carriers_ghz: RealArray
    qubit_carriers_ghz: RealArray
    qubit_frequency_ghz: float
    incident_per_sqrt_ns: float
    down_per_ns: float
    up_per_ns: float
    gamma2_per_ns: float
    pi_duration_ns: float
    pi_rabi_rad_per_ns: float
    prefill_duration_ns: float
    probe_duration_ns: float
    probe_rabi_rad_per_ns: float

    def __post_init__(self) -> None:
        for name, shape in (
            ("frequencies_ghz", (2, 2)),
            ("kappa_external_per_ns", (2,)),
            ("kappa_internal_per_ns", (2,)),
        ):
            value = _owned(getattr(self, name), name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if name.startswith("kappa") and np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        for name in ("resonator_carriers_ghz", "qubit_carriers_ghz"):
            value = _owned(getattr(self, name), name)
            if value.ndim != 1 or not value.size:
                raise ValueError(f"{name} must be a nonempty vector")
            object.__setattr__(self, name, value)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                continue
            if not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if (
                field.name.endswith("_per_ns") or field.name.endswith("_duration_ns")
            ) and value < 0:
                raise ValueError(f"{field.name} must be nonnegative")
        if self.gamma2_per_ns < (self.down_per_ns + self.up_per_ns) / 2:
            raise ValueError("gamma2_per_ns implies negative pure dephasing")
        if self.up_per_ns * self.prefill_duration_ns > 1e-12:
            raise ValueError("One-jump prefill requires up * prefill duration <= 1e-12")


@dataclass(frozen=True)
class JointFockResult:
    """Raw populations and solver diagnostics, never clipped or renormalized.

    populations: (rf, preparation, qf, state), with preparation ground then
        finite-pi and state g then e. Values are cavity partial traces.
    cutoffs: one total cavity excitation N per rf, after conservative N+1.
    trace_error, hermitian_error, min_eigenvalue, edge_population: real arrays
        (rf, preparation, qf) of final joint density trace-minus-one magnitude,
        maximum Hermitian defect, minimum Hermitianized eigenvalue, and
        population on n1+n2=N. These are observations, not error certificates.
    """

    populations: RealArray
    cutoffs: tuple[int, ...]
    trace_error: RealArray
    hermitian_error: RealArray
    min_eigenvalue: RealArray
    edge_population: RealArray


def _owned(value: RealArray, name: str) -> RealArray:
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.flags.writeable = False
    return result


def _commutator(h: sparse.csr_matrix) -> sparse.csr_matrix:
    identity = sparse.csr_matrix(np.eye(len(h.indptr) - 1, dtype=np.complex128))
    return sparse.csr_matrix(
        -1j * (sparse.kron(identity, h) - sparse.kron(h.T, identity))
    )


def _dissipator(op: sparse.csr_matrix) -> sparse.csr_matrix:
    identity = sparse.csr_matrix(np.eye(len(op.indptr) - 1, dtype=np.complex128))
    square = op.conj().T @ op
    return sparse.csr_matrix(
        sparse.kron(op.conj(), op)
        - (sparse.kron(identity, square) + sparse.kron(square.T, identity)) / 2
    )


def _nominal_pi(request: JointFockRequest) -> ComplexArray:
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


def _rk4(
    generator: sparse.csr_matrix, states: ComplexArray, duration: float
) -> ComplexArray:
    result = states.copy()
    steps = max(1, int(np.ceil(duration)))
    dt = duration / steps
    for _ in range(steps):
        k1 = generator @ result
        k2 = generator @ (result + dt / 2 * k1)
        k3 = generator @ (result + dt / 2 * k2)
        k4 = generator @ (result + dt * k3)
        result += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite joint-Fock RK4 density")
    return result


def _vacuum_row(request: JointFockRequest) -> ComplexArray:
    sm = sparse.csr_matrix([[0, 1], [0, 0]], dtype=complex)
    pe = sparse.diags([0, 1], format="csr", dtype=complex)
    z = sparse.csr_matrix(np.diag([1.0, -1.0]))
    bath = request.down_per_ns * _dissipator(sm) + request.up_per_ns * _dissipator(
        sm.conj().T.tocsr()
    )
    bath += (
        (request.gamma2_per_ns - (request.down_per_ns + request.up_per_ns) / 2)
        / 2
        * _dissipator(z)
    )
    states = np.column_stack(
        [np.diag([1, 0]).ravel(order="F"), _nominal_pi(request).ravel(order="F")]
    ).astype(complex)
    states = np.asarray(
        expm(bath.toarray() * request.prefill_duration_ns) @ states, dtype=np.complex128
    )
    angle = (
        2
        * np.pi
        * request.qubit_frequency_ghz
        * (request.pi_duration_ns + request.prefill_duration_ns)
    )
    rotation = np.array([np.exp(0.5j * angle), np.exp(-0.5j * angle)])
    states *= (rotation[:, None] * rotation.conj()[None, :]).ravel(order="F")[:, None]
    outputs = []
    for qf in request.qubit_carriers_ghz:
        h = 2 * np.pi * (
            request.qubit_frequency_ghz - qf
        ) * pe + request.probe_rabi_rad_per_ns / 2 * (sm + sm.conj().T)
        output = _rk4(
            (bath + _commutator(h)).tocsr(), states, request.probe_duration_ns
        )
        outputs.append(
            np.stack([output[:, b].reshape(2, 2, order="F") for b in range(2)])
        )
    return np.stack(outputs, axis=1)


def joint_fock_spectrum(request: JointFockRequest) -> JointFockResult:
    """Compute an explicit joint spectrum; does not select an acquisition backend.

    Takes only JointFockRequest. Returns raw final populations and density
    diagnostics for each independent rf/preparation/qf. Constant generators use
    fixed RK4 steps <= 1 ns; no positivity or global error bound is implied.
    No state survives the call. Unsupported driven cavities raise ValueError.
    """
    if request.incident_per_sqrt_ns != 0:
        raise ValueError("Driven joint cavities are not yet supported")
    row = _vacuum_row(request)
    densities = np.repeat(row[None, ...], len(request.resonator_carriers_ghz), axis=0)
    populations = densities.diagonal(axis1=-2, axis2=-1).real.copy()
    shape = populations.shape[:-1]
    return JointFockResult(
        populations=populations,
        cutoffs=tuple(1 for _ in request.resonator_carriers_ghz),
        trace_error=abs(np.trace(densities, axis1=-2, axis2=-1) - 1),
        hermitian_error=np.max(
            abs(densities - densities.conj().swapaxes(-1, -2)), axis=(-1, -2)
        ),
        min_eigenvalue=np.linalg.eigvalsh(
            (densities + densities.conj().swapaxes(-1, -2)) / 2
        ).min(axis=-1),
        edge_population=np.zeros(shape),
    )
