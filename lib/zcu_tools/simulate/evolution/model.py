"""Fixed-coordinate numeric model records, separate from Engine model sources."""

from dataclasses import dataclass

import numpy as np

from .state import Basis, ComplexMatrix, owned_matrix


def hermitian_matrix(value: ComplexMatrix, name: str) -> ComplexMatrix:
    """Own and validate a finite Hermitian complex matrix (tolerance 1e-10)."""
    matrix = owned_matrix(value, name)
    if not np.allclose(matrix, matrix.conj().T, rtol=0, atol=1e-10):
        raise ValueError(f"{name} must be Hermitian")
    return matrix


@dataclass(frozen=True)
class DriveSource:
    """Numeric quadrature operators for one physical source in the model basis.

    name: unique nonempty source identity used by controls, not a QICK channel.
    in_phase_ghz: Hermitian H/h contribution in GHz per real envelope amplitude.
    quadrature_ghz: Hermitian H/h contribution in GHz per imaginary amplitude;
        None means zero. H/h += Re(z)*in_phase + Im(z)*quadrature, where z is the
        complete envelope/carrier/frame. No hidden 1/2, 2π or gain rescaling.
    Both matrices own read-only copies. Invalid values raise ValueError.
    """

    name: str
    in_phase_ghz: ComplexMatrix
    quadrature_ghz: ComplexMatrix | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("source requires a name")
        x = hermitian_matrix(self.in_phase_ghz, "source in-phase operator")
        y = np.zeros_like(x) if self.quadrature_ghz is None else self.quadrature_ghz
        y = hermitian_matrix(y, "source quadrature operator")
        if y.shape != x.shape:
            raise ValueError("source quadrature shapes must match")
        object.__setattr__(self, "in_phase_ghz", x)
        object.__setattr__(self, "quadrature_ghz", y)


@dataclass(frozen=True)
class NumericModel:
    """Fixed-coordinate numeric model; no device, YAML, FTF or QICK dependency.

    identity: nonempty reproducible caller-assigned model/version identifier.
    basis: fixed state coordinates/order for every operator and density.
    hamiltonian_ghz: static Hermitian H/h in GHz, including any DC couplings.
    sources: named quadrature operators in basis. Names must be unique.
    collapse_operators: Lindblad L matrices in ns**(-1/2), including rates through
        their amplitudes; may describe directional multilevel jumps/dephasing.
    event_times_ns: finite absolute event coordinates in ns, default empty.
        Metadata only; this record does not partition or perform evolution.
    Matrices own read-only copies. Invalid identity, shape, finiteness or
    Hermiticity raises ValueError at construction. Engine consumes ModelSource
    and SecularModel rather than this fixed-coordinate record.
    No physical applicability/convergence certification is performed.
    """

    identity: str
    basis: Basis
    hamiltonian_ghz: ComplexMatrix
    sources: tuple[DriveSource, ...] = ()
    collapse_operators: tuple[ComplexMatrix, ...] = ()
    event_times_ns: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.event_times_ns).all():
            raise ValueError("model event times must be finite ns")
        h = hermitian_matrix(self.hamiltonian_ghz, "Hamiltonian")
        if not self.identity or h.shape != self.basis.vectors.shape:
            raise ValueError("model identity and Hamiltonian shape must match basis")
        if len({s.name for s in self.sources}) != len(self.sources):
            raise ValueError("source names must be unique")
        if any(s.in_phase_ghz.shape != h.shape for s in self.sources):
            raise ValueError("source shapes must match model basis")
        collapse = tuple(
            owned_matrix(c, "collapse operator") for c in self.collapse_operators
        )
        if any(c.shape != h.shape for c in collapse):
            raise ValueError("collapse shapes must match model basis")
        object.__setattr__(self, "hamiltonian_ghz", h)
        object.__setattr__(self, "collapse_operators", collapse)
