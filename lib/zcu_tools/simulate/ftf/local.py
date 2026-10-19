"""Fixed native/local circuit operators; no moving pulse basis."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cosm, sinm

from .parameters import FluxoniumCircuit, SquidCircuit

Matrix = NDArray[np.complex128]


@dataclass(frozen=True)
class LocalCircuit:
    """Internal fixed-DC local projection of native physical operators."""

    parameters: FluxoniumCircuit | SquidCircuit
    vectors: Matrix
    charge: Matrix
    charge_squared: Matrix
    phase: Matrix
    cosine: Matrix
    sine: Matrix
    cos_half: Matrix
    sin_half: Matrix
    kinetic: Matrix

    def hamiltonian(self, flux: float) -> Matrix:
        p = self.parameters
        if isinstance(p, FluxoniumCircuit):
            angle = 2 * np.pi * flux
            return self.kinetic - p.ej_ghz * (
                np.cos(angle) * self.cosine - np.sin(angle) * self.sine
            )
        angle = np.pi * flux
        return (
            self.kinetic
            - p.ejmax_ghz * np.cos(angle) * self.cosine
            + p.asymmetry * p.ejmax_ghz * np.sin(angle) * self.sine
        )

    def derivative(self, flux: float) -> Matrix:
        p = self.parameters
        if isinstance(p, FluxoniumCircuit):
            angle = 2 * np.pi * flux
            return (
                2
                * np.pi
                * p.ej_ghz
                * (np.sin(angle) * self.cosine + np.cos(angle) * self.sine)
            )
        angle = np.pi * flux
        return (
            np.pi
            * p.ejmax_ghz
            * (np.sin(angle) * self.cosine + p.asymmetry * np.cos(angle) * self.sine)
        )

    def quasiparticle(self, flux: float) -> Matrix:
        return (
            np.sin(np.pi * flux) * self.cos_half + np.cos(np.pi * flux) * self.sin_half
        )


@lru_cache(maxsize=8)
def _fluxonium(ec_ghz: float, el_ghz: float, n: int) -> tuple[Matrix, ...]:
    # Compute n² before truncation; retain the missing raising matrix element
    # at the top oscillator level instead of squaring an already cut n.
    a = np.diag(np.sqrt(np.arange(1, n + 1)), 1).astype(complex)
    phi_zpf = (2 * ec_ghz / el_ghz) ** 0.25
    phi = phi_zpf * (a + a.conj().T)
    charge = 1j * (a.conj().T - a) / (2 * phi_zpf)
    n2 = (charge @ charge)[:n, :n]
    phi, charge = phi[:n, :n], charge[:n, :n]
    lc = np.diag(np.sqrt(8 * ec_ghz * el_ghz) * (np.arange(n) + 0.5)).astype(complex)
    operators = tuple(
        np.asarray(op, dtype=np.complex128)
        for op in (
            charge,
            n2,
            phi,
            cosm(phi),
            sinm(phi),
            cosm(phi / 2),
            sinm(phi / 2),
            lc,
        )
    )
    for operator in operators:
        operator.setflags(write=False)
    return operators


def _squid(p: SquidCircuit) -> tuple[Matrix, ...]:
    numbers = np.arange(-p.charge_cutoff, p.charge_cutoff + 1)
    n = len(numbers)
    charge = np.diag(numbers).astype(complex)
    shift = np.eye(n, k=1, dtype=complex)
    cosine = (shift + shift.conj().T) / 2
    sine = (shift - shift.conj().T) / (2j)
    zero = np.zeros_like(charge)
    kinetic = np.diag(4 * p.ec_ghz * (numbers - p.ng) ** 2).astype(complex)
    return tuple(
        np.asarray(op, dtype=np.complex128)
        for op in (charge, charge @ charge, zero, cosine, sine, zero, zero, kinetic)
    )


def local_circuit(
    p: FluxoniumCircuit | SquidCircuit, flux: float
) -> tuple[LocalCircuit, NDArray[np.float64]]:
    """Build native then fixed-DC local operators, returning retained energies."""
    operators = (
        _fluxonium(p.ec_ghz, p.el_ghz, p.native_cutoff)
        if isinstance(p, FluxoniumCircuit)
        else _squid(p)
    )
    native = LocalCircuit(p, np.eye(len(operators[0]), dtype=complex), *operators)
    energies, vectors = np.linalg.eigh(native.hamiltonian(flux))
    vectors = vectors[:, : p.levels]
    projected = tuple(vectors.conj().T @ op @ vectors for op in operators)
    return LocalCircuit(p, vectors, *projected), energies[: p.levels]


def tensor_operator(
    operator: Matrix, index: int, dimensions: tuple[int, ...]
) -> Matrix:
    """Lift a local operator into the F1-C-F2 product coordinate space."""
    result = np.ones((1, 1), dtype=complex)
    for i, dimension in enumerate(dimensions):
        result = np.kron(result, operator if i == index else np.eye(dimension))
    return np.asarray(result, dtype=np.complex128)
