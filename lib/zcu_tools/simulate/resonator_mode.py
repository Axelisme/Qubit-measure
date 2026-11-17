"""Add a quantum resonator to a full native-cutoff qubit Hamiltonian."""

# ADR-0072: numeric composition belongs in simulate, not a physical model adapter.

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .evolution import FluxLevel, SecularDrive, SecularModel, digest


@dataclass(frozen=True)
class ResonatorModeInput:
    """Numeric inputs for one qubit plus one Fock-truncated resonator.

    hamiltonian_ghz: full native-cutoff Hermitian qubit H/h, GHz, fixed coordinates.
    coupling_ghz: Hermitian sum of charge couplings, GHz, in the same coordinates;
        the resonator factor is a + a†, with no rotating-wave approximation.
    drive_operators_ghz: port name to Hermitian qubit H/h per real envelope unit,
        in the same coordinates. Imaginary-envelope quadratures are zero.
    retained_labels: distinct bare (q, k) pairs to keep; q indexes ascending parent
        eigenstates and k indexes dimensionless photon occupation, 0 <= k < N.
    frequency_ghz: finite positive resonator frequency in cycles/ns (GHz).
    decay_per_ns: finite nonnegative resonator energy-loss rate kappa in 1/ns,
        not kappa/(2*pi).
    qubit_rates_per_ns: finite nonnegative parent eigenbasis R[target, source],
        shape (Q,Q), 1/ns, zero diagonal; upward parent rates are allowed.
    level: quantized flux tuple in the caller's loop order (FluxLevel).
    photon_cutoff: positive number N of photon states, default 3 (0..N-1).
    """

    hamiltonian_ghz: NDArray[np.complex128]
    coupling_ghz: NDArray[np.complex128]
    drive_operators_ghz: dict[str, NDArray[np.complex128]]
    retained_labels: tuple[tuple[int, int], ...]
    frequency_ghz: float
    decay_per_ns: float
    qubit_rates_per_ns: NDArray[np.float64]
    level: FluxLevel
    photon_cutoff: int = 3


@dataclass(frozen=True)
class ResonatorModeResult:
    """Dressed model with bare-state names for initial states and transitions.

    model: joint SecularModel in ascending dressed energy order, GHz and 1/ns;
        fixed coordinates are this dressed basis (eigenvectors are identity).
    label_to_index: bare (parent eigenstate, photon occupation) to output index.
    index_to_label: inverse map, indexed by output index.
    """

    model: SecularModel
    label_to_index: dict[tuple[int, int], int]
    index_to_label: tuple[tuple[int, int], ...]


def _validate_operator(operator: NDArray[np.complex128], size: int, name: str) -> None:
    """Reject nonfinite, nonsquare or non-Hermitian native qubit operators."""
    if operator.shape != (size, size) or not np.isfinite(operator).all():
        raise ValueError(f"{name} must be a finite ({size}, {size}) Hermitian matrix")
    if not np.allclose(operator, operator.conj().T, rtol=0, atol=1e-10):
        raise ValueError(f"{name} must be Hermitian")


def _validate_inputs(inputs: ResonatorModeInput) -> None:
    """Validate the native-coordinate numeric contract before diagonalization."""
    if inputs.hamiltonian_ghz.ndim != 2 or not inputs.hamiltonian_ghz.size:
        raise ValueError("parent must be a nonempty Hermitian matrix")
    size = inputs.hamiltonian_ghz.shape[0]
    _validate_operator(inputs.hamiltonian_ghz, size, "parent")
    _validate_operator(inputs.coupling_ghz, size, "coupling")
    for name, operator in inputs.drive_operators_ghz.items():
        if not name:
            raise ValueError("drive port name must be nonempty")
        _validate_operator(operator, size, f"drive {name}")
    rates = inputs.qubit_rates_per_ns
    if (
        rates.shape != (size, size)
        or not np.isfinite(rates).all()
        or np.any(rates < 0)
        or np.any(np.diag(rates) != 0)
    ):
        raise ValueError(
            "qubit rates must be finite nonnegative (Q,Q) with zero diagonal"
        )
    if inputs.photon_cutoff < 1:
        raise ValueError("photon cutoff must be positive")
    if not np.isfinite(inputs.frequency_ghz) or inputs.frequency_ghz <= 0:
        raise ValueError("resonator frequency must be finite and positive")
    if not np.isfinite(inputs.decay_per_ns) or inputs.decay_per_ns < 0:
        raise ValueError("resonator decay must be finite and nonnegative")
    if not inputs.retained_labels:
        raise ValueError("retained labels must be nonempty")
    for q, k in inputs.retained_labels:
        if not (0 <= q < size and 0 <= k < inputs.photon_cutoff):
            raise ValueError(
                f"label {(q, k)} has overlap 0: outside parent/photon cutoff"
            )


def _select_labels(
    overlaps: NDArray[np.float64], labels: tuple[tuple[int, int], ...], cutoff: int
) -> dict[int, tuple[int, int]]:
    """Select energy indices with a unique strict-majority bare label."""
    selected: dict[int, tuple[int, int]] = {}
    for label in labels:
        q, k = label
        row = overlaps[q * cutoff + k]
        index = int(np.argmax(row))
        overlap = float(row[index])
        if overlap <= 0.5 or index in selected:
            raise ValueError(
                f"label {label} has nonunique or insufficient overlap {overlap:.16g}"
            )
        selected[index] = label
    return selected


def build_resonator_mode(inputs: ResonatorModeInput) -> ResonatorModeResult:
    """Build a joint dressed model without mutating inputs.

    Each retained label must uniquely dominate a dressed state with bare overlap
    probability strictly greater than 0.5. Otherwise raise ValueError naming the
    label and overlap. Invalid shapes, non-Hermitian operators or invalid rates
    also raise ValueError. Resonator jumps only lower dressed energy; parent
    qubit loss channels are projected into the joint basis. Pure dephasing is zero.
    """
    _validate_inputs(inputs)
    parent_energies, parent_vectors = np.linalg.eigh(inputs.hamiltonian_ghz)
    photons = np.eye(inputs.photon_cutoff)
    annihilation = np.diag(np.sqrt(np.arange(1, inputs.photon_cutoff)), 1)
    hamiltonian = (
        np.kron(inputs.hamiltonian_ghz, photons)
        + np.kron(
            np.eye(len(parent_energies)),
            inputs.frequency_ghz * annihilation.T @ annihilation,
        )
        + np.kron(inputs.coupling_ghz, annihilation + annihilation.T)
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    bare_vectors = np.kron(parent_vectors, photons)
    overlaps = np.abs(bare_vectors.conj().T @ vectors) ** 2
    selected = _select_labels(overlaps, inputs.retained_labels, inputs.photon_cutoff)
    indices = sorted(selected)
    labels = tuple(selected[i] for i in indices)
    retained = vectors[:, indices]
    drives = tuple(
        SecularDrive(
            name,
            np.asarray(
                retained.conj().T @ np.kron(operator, photons) @ retained,
                dtype=np.complex128,
            ),
            np.zeros((len(indices), len(indices)), dtype=complex),
        )
        for name, operator in sorted(inputs.drive_operators_ghz.items())
    )
    projected_a = (
        retained.conj().T
        @ np.kron(np.eye(len(parent_energies)), annihilation)
        @ retained
    )
    energies = energies[indices]
    rates = inputs.decay_per_ns * np.abs(projected_a) ** 2
    rates *= energies[:, None] < energies[None, :]
    for target, source in zip(*np.nonzero(inputs.qubit_rates_per_ns), strict=True):
        jump = np.outer(parent_vectors[:, target], parent_vectors[:, source].conj())
        projected_jump = retained.conj().T @ np.kron(jump, photons) @ retained
        rates += inputs.qubit_rates_per_ns[target, source] * np.abs(projected_jump) ** 2
    np.fill_diagonal(rates, 0)
    model = SecularModel(
        level=inputs.level,
        identity=digest(asdict(inputs)),
        energies_ghz=energies,
        eigenvectors=np.eye(len(indices), dtype=complex),
        drives=drives,
        rates_per_ns=rates,
        dephasing_per_ns=np.zeros_like(rates),
    )
    return ResonatorModeResult(
        model, {label: i for i, label in enumerate(labels)}, labels
    )
