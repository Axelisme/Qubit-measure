"""Model lookup, drive ratios and modal packing for Dyson and its B2 bound."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..secular import ModelSource, SecularModel
from ._secular import coherence_decay, population_generator, relaxation_decay


def model_keys(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.void]:
    """Encode flux levels; the engine supplies the round's DC-model namespace."""
    keys = np.empty(
        len(rows), dtype=[("level", "i8", table.level.shape[1]), ("shift", "S64")]
    )
    keys["level"] = table.level[rows]
    keys["shift"] = table.shift_content[rows]
    return keys


def model_groups(
    table: SegmentTable, rows: NDArray[np.int64], source: ModelSource
) -> Iterator[tuple[NDArray[np.int64], SecularModel]]:
    """Yield row positions and models, looping only over unique flux levels."""
    _, first, inverse = np.unique(
        model_keys(table, rows), return_index=True, return_inverse=True
    )
    for index, representative in enumerate(first):
        yield (
            np.flatnonzero(inverse == index),
            table.model(source, int(rows[representative])),
        )


def constant_mask(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.bool_]:
    """One playing tone with a unity envelope is the Dyson domain."""
    result = table.tones[rows] == 1
    if table.tone_port.shape[1] == 0:
        return np.zeros(len(rows), dtype=np.bool_)
    constant = np.array([env.kind == "constant" for env in table.envelopes])
    active = np.flatnonzero(result)
    result[active] &= constant[table.tone_envelope[rows[active], 0]]
    return result


def generator(model: SecularModel) -> NDArray[np.complex128]:
    """Return L0 on row-major flattened density, in inverse ns."""
    size = len(model.energies_ghz)
    # Keep the two real losses separate: combining them changes rounding of
    # the complex diagonal and therefore the modal eigensystem.
    diagonal = (
        -2j * np.pi * (model.energies_ghz[:, None] - model.energies_ghz[None, :])
        - relaxation_decay(model.rates_per_ns)
        - model.dephasing_per_ns
    )
    result = np.asarray(np.diag(diagonal.ravel()), dtype=np.complex128)
    populations = np.arange(size) * (size + 1)
    result[np.ix_(populations, populations)] = population_generator(model.rates_per_ns)
    return result


def commutator(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return -2πi[A, .] in row-major coordinates."""
    identity = np.eye(len(matrix))
    return -2j * np.pi * (np.kron(matrix, identity) - np.kron(identity, matrix.T))


def parameters(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
    """Pack array-backend columns f (GHz), gain, duration (ns), phase (rad)."""
    return np.column_stack(
        (
            table.tone_frequency_ghz[rows, 0],
            table.tone_amplitude[rows, 0],
            table.duration_ns[rows],
            table.tone_theta_rad[rows, 0],
        )
    )


def drive_ratios(
    model: SecularModel, port: int, params: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Finite-duration P8 one/two-photon ratios, independent of the input state.

    Width is sqrt(gamma_ij**2 + duration**-2). The one-photon numerator
    is 2*abs(gain)*max(abs(A+_ij), abs(A-_ij)), A±=(X∓iY)/2.
    Two-photon paths use max(abs(Ek-Ei-f), duration**-1) denominators.
    A zero-duration row has both ratios zero.
    """
    frequency, gain, duration = abs(params[:, 0]), abs(params[:, 1]), params[:, 2]
    inverse_time = np.full(len(params), np.inf)
    np.divide(1.0, duration, out=inverse_time, where=duration > 0)
    drive = model.drives[port]
    plus = (drive.in_phase_ghz - 1j * drive.quadrature_ghz) / 2
    minus = plus.conj().T
    energies = model.energies_ghz
    decay = coherence_decay(model)
    one = np.zeros(len(params))
    two = np.zeros(len(params))
    for i in range(len(energies)):
        for j in range(i + 1, len(energies)):
            gap = energies[j] - energies[i]
            gamma = decay[i, j]
            width2 = gamma**2 + inverse_time**2
            strength = 2 * gain * max(abs(plus[i, j]), abs(minus[i, j]))
            one = np.maximum(one, strength / np.sqrt((frequency - gap) ** 2 + width2))
            amplitude = np.zeros(len(params))
            for k in range(len(energies)):
                amplitude += (
                    gain**2
                    * max(abs(plus[i, k] * plus[k, j]), abs(minus[i, k] * minus[k, j]))
                    / np.maximum(
                        abs(energies[k] - energies[i] - frequency), inverse_time
                    )
                )
            two = np.maximum(
                two, amplitude / np.sqrt((2 * frequency - gap) ** 2 + width2)
            )
    return one, two


def freeze(array: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Make an owned contiguous numeric payload read-only before publication."""
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result
