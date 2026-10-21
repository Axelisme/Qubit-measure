"""Cutoff-aware native oscillator coordinates for a single fluxonium."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def validate_circuit(params: tuple[float, float, float], cutoff: int) -> None:
    """Validate (EJ, EC, EL) GHz and oscillator dimension; raise ValueError.

    EJ is finite nonnegative (zero admits an LC reference); EC/EL are positive.
    cutoff is a Python integer >=2. No circuit or convergence calculation occurs.
    """
    if (
        len(params) != 3
        or not np.isfinite(params).all()
        or params[0] < 0
        or min(params[1:]) <= 0
    ):
        raise ValueError("circuit requires finite EJ >= 0 and EC, EL > 0 in GHz")
    if type(cutoff) is not int or cutoff < 2:
        raise ValueError("oscillator cutoff must be an integer >= 2")


@dataclass(frozen=True)
class FluxoniumOperators:
    """Read-only oscillator-basis operators; all matrices have shape (cutoff,cutoff).

    static_ghz: diagonal LC H/h, GHz. cosine_ghz: -EJ*cos(phi), GHz.
    sine_ghz: +EJ*sin(phi), GHz. charge: dimensionless Cooper-pair n, complex128.
    The exact native H(f)/h is static_ghz + cos(2*pi*f)*cosine_ghz
    + sin(2*pi*f)*sine_ghz, for absolute reduced flux f=Phi/Phi0.
    Matrices are owned immutable copies; invalid square/Hermitian/finite data
    or unequal dimensions raise ValueError. Native cutoff is not convergence.
    """

    static_ghz: ComplexArray
    cosine_ghz: ComplexArray
    sine_ghz: ComplexArray
    charge: ComplexArray

    def __post_init__(self) -> None:
        expected = np.asarray(self.static_ghz).shape
        if len(expected) != 2 or expected[0] != expected[1] or expected[0] < 2:
            raise ValueError("operators require square oscillator dimension >= 2")
        for name in ("static_ghz", "cosine_ghz", "sine_ghz", "charge"):
            matrix = np.array(getattr(self, name), dtype=np.complex128, copy=True)
            if (
                matrix.shape != expected
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix, matrix.conj().T, rtol=0, atol=1e-12)
            ):
                raise ValueError(f"{name} must be finite Hermitian shape {expected}")
            matrix.setflags(write=False)
            object.__setattr__(self, name, matrix)


def calculate_oscillator_operators(
    params: tuple[float, float, float], *, cutoff: int
) -> FluxoniumOperators:
    """Return native LC/cos/sin/charge operators for (EJ,EC,EL) GHz.

    cutoff sets the oscillator dimension, independent of retained energy levels.
    See FluxoniumOperators for signs, units and reconstruction. EJ >=0, EC/EL >0
    and integer cutoff >=2 are required; invalid input raises ValueError.
    scqubits loads lazily and numerical failures propagate. No global settings
    or caller-owned arrays change.
    """
    from scqubits.core.fluxonium import Fluxonium

    validate_circuit(params, cutoff)
    circuit = Fluxonium(*params, flux=0.0, cutoff=cutoff, truncated_dim=cutoff)
    return FluxoniumOperators(
        np.diag((np.arange(cutoff) + 0.5) * circuit.plasma_energy()).astype(complex),
        np.asarray(-params[0] * circuit.cos_phi_operator(), dtype=np.complex128),
        np.asarray(params[0] * circuit.sin_phi_operator(), dtype=np.complex128),
        np.asarray(circuit.n_operator(), dtype=np.complex128),
    )


def calculate_eigensystem(
    params: tuple[float, float, float], flux: float, *, cutoff: int, evals_count: int
) -> tuple[RealArray, ComplexArray]:
    """Return native energies GHz (M,) and oscillator energy columns (cutoff,M).

    params is (EJ,EC,EL) GHz as in calculate_oscillator_operators; flux is finite
    absolute Phi/Phi0 (not folded, so vector phases/coordinates stay meaningful).
    evals_count=M is an integer in [1,cutoff]. Results own read-only storage,
    ascending energies and orthonormal columns. Each call diagonalizes the
    native Hamiltonian independently. Invalid inputs raise ValueError; scqubits
    numerical failures propagate, without changing global settings.
    """
    from scqubits.core.fluxonium import Fluxonium

    validate_circuit(params, cutoff)
    if not np.isfinite(flux):
        raise ValueError("flux must be finite Phi/Phi0")
    if type(evals_count) is not int or not 1 <= evals_count <= cutoff:
        raise ValueError("evals_count must be an integer in [1,cutoff]")
    circuit = Fluxonium(*params, flux=flux, cutoff=cutoff, truncated_dim=evals_count)
    energy, vectors = circuit.eigensys(evals_count=evals_count)
    energy = np.array(energy, dtype=np.float64, copy=True)
    vectors = np.array(vectors, dtype=np.complex128, copy=True)
    energy.setflags(write=False)
    vectors.setflags(write=False)
    return energy, vectors
