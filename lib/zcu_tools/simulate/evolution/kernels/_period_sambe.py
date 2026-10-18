"""Bounded spectral construction of constant-tone physical Hilbert endpoints."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..secular import SecularModel
from ._sambe import eigensystem


class Spectrum(NamedTuple):
    """Immutable spectral trajectory, with energies/frequency in signed GHz.

    values and vectors diagonalize the truncated Sambe matrix. central is its
    central-input projection, modes is the retained harmonic cutoff, and
    frequency is the key's original signed carrier. All arrays are read-only.
    """

    values: NDArray[np.float64]
    vectors: NDArray[np.complex128]
    central: NDArray[np.complex128]
    modes: int
    frequency: float

    @property
    def nbytes(self) -> int:
        return self.values.nbytes + self.vectors.nbytes + self.central.nbytes


@njit(nogil=True, cache=True)
def endpoint(spectrum: Spectrum, time: float) -> NDArray[np.complex128]:
    """Reconstruct physical U(time) from a central harmonic input, time in ns."""
    values, vectors, central, modes, frequency = spectrum
    size = central.shape[1]
    evolved = (vectors * np.exp(-2j * np.pi * values * time)) @ central
    result = np.zeros((size, size), dtype=np.complex128)
    for index in range(2 * modes + 1):
        result += (
            np.exp(2j * np.pi * (index - modes) * frequency * time)
            * evolved[index * size : (index + 1) * size]
        )
    return result


@njit(nogil=True, cache=True)
def _probes(spectrum):
    size = spectrum.central.shape[1]
    result = np.empty((9, size, size), dtype=np.complex128)
    for j in range(9):
        result[j] = endpoint(spectrum, j / (8 * abs(spectrum.frequency)))
    return result


def build(
    model: SecularModel,
    port: int,
    frequency: float,
    gain: float,
    budget: float,
    work_limit: float,
) -> tuple[Spectrum, NDArray[np.complex128]] | None:
    """Return a converged spectrum/U(T), or None for independent CF4 fallback.

    Compare adjacent cutoffs at q=0,1/8,...,1 with Frobenius defect no larger
    than min(1e-4,budget*0.1), retaining the finer cutoff. Cumulative eigensolve
    work cannot exceed dispatch's supplied CF4 work_limit (N-cubed units),
    with an absolute cutoff of 12.
    Nonfinite or non-near-unitary probes and failed eigensolves decline too.
    This calibrated nine-point check assumes cutoff convergence and that samples
    represent between-node error; it is not an all-time operator error bound.
    """
    drive = model.drives[port]
    coupling = gain * (drive.in_phase_ghz - 1j * drive.quadrature_ghz) / 2
    energies = model.energies_ghz - model.energies_ghz[0]
    spent = 0.0
    previous = None
    for modes in range(2, 13):
        spent += 0.5 * (2 * modes + 1) ** 3
        if spent > work_limit:
            break
        try:
            values, vectors, central = eigensystem(energies, coupling, frequency, modes)
            data = values, vectors, np.ascontiguousarray(central.conj().T)
        except np.linalg.LinAlgError:
            return None
        if not all(np.isfinite(array).all() for array in data):
            return None
        spectrum = Spectrum(*data, modes, frequency)
        probes = _probes(spectrum)
        if not np.isfinite(probes).all():
            return None
        if previous is not None:
            error = max(
                np.linalg.norm(a - b) for a, b in zip(probes, previous, strict=True)
            )
            defect = max(
                np.linalg.norm(a.conj().T @ a - np.eye(len(a))) for a in probes
            )
            if error <= min(1e-4, budget * 0.1) and defect <= 1e-4:
                for array in data:
                    array.setflags(write=False)
                return spectrum, probes[-1]
        previous = probes
    return None
