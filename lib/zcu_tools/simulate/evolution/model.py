"""Named pure-numeric Hamiltonian and solver settings."""

from collections.abc import Callable
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
    hamiltonian_at: optional pure callable of absolute ns returning an additive
        Hermitian H/h matrix in GHz, in the same fixed basis (e.g. dynamic flux).
    collapse_at: optional pure callable of absolute ns returning additional
        Lindblad operators in ns**(-1/2), e.g. flux-dependent baths. Caller must
        keep callbacks deterministic and unchanged while reusing this model.
    Static matrices own read-only copies. Callbacks remain caller-owned. Shape,
    finiteness or Hermiticity violations raise ValueError, including at runtime.
    No physical applicability/convergence certification is performed.
    """

    identity: str
    basis: Basis
    hamiltonian_ghz: ComplexMatrix
    sources: tuple[DriveSource, ...] = ()
    collapse_operators: tuple[ComplexMatrix, ...] = ()
    hamiltonian_at: Callable[[float], ComplexMatrix] | None = None
    collapse_at: Callable[[float], tuple[ComplexMatrix, ...]] | None = None

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class SolverSettings:
    """Direct-solver accuracy controls, not a physical accuracy guarantee.

    rtol/atol: positive finite DOP853 relative/absolute local error tolerances.
    max_step_ns: positive maximum internal integration step in ns; infinity lets
        the adaptive solver choose. This is unrelated to the 2 ns event grid.
    Closed static intervals use expm. Closed single/commensurate-frequency flat
    intervals may use one exact-frequency period and a tail, with binary powers.
    Dissipative intervals always solve the full Lindblad equation; no closed
    propagator substitutes for a bath. No offline persistence or fake backend.
    """

    rtol: float = 1e-10
    atol: float = 1e-12
    max_step_ns: float = np.inf

    def __post_init__(self) -> None:
        if (
            not np.isfinite([self.rtol, self.atol]).all()
            or min(self.rtol, self.atol) <= 0
        ):
            raise ValueError("solver tolerances must be positive finite")
        if np.isnan(self.max_step_ns) or self.max_step_ns <= 0:
            raise ValueError("maximum solver step must be positive")
