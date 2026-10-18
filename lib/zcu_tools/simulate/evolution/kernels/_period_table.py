"""Shared immutable period tables and single-threaded envelope playback."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ..kernel import SegmentTable
from ..workplan import WorkPlan
from ._dop853_data import pack_segments
from ._workload_facts import envelope_bounds
from .dop853 import DOP853Plan, DOP853Prepared, advance_endpoints


class PeriodPlayback(NamedTuple):
    """Per-row carrier playback, aligned with mutable states (n,M,M).

    ports: (n,) local indices into UnitaryTable.coefficients' first axis.
    amplitudes: (n,K) complex midpoint gains, K >= max(counts).
    phases: (n,) carrier phases at row start, in rad.
    counts: (n,) nonnegative complete-subperiod counts.
    Arrays are borrowed for this call; playback does not mutate them.
    """

    ports: NDArray[np.int64]
    amplitudes: NDArray[np.complex128]
    phases: NDArray[np.float64]
    counts: NDArray[np.int64]


class UnitaryTable(NamedTuple):
    """Immutable interpolation and carrier lattice for period playback.

    bound: positive absolute gain ceiling for the Chebyshev interval [0,bound].
    coefficients: (P,D,16,M,M) amplitude/Fourier coefficients, 1 <= D <= 8.
    subdivisions: positive subperiod count per carrier period.
    block_cycles: positive maximum block length; one for subdivided periods.
    project_unitary: whether to apply two Newton polar iterations to U.
    direction: +1 or -1 for signed carrier phase advancement.
    harmonics: retained absolute Fourier cutoff, 0..8; 8 keeps all 16 slots.
    """

    bound: float
    coefficients: NDArray[np.complex128]
    subdivisions: int
    block_cycles: int
    project_unitary: bool
    direction: int
    harmonics: int


class SecularMaps(NamedTuple):
    """Borrowed (M,M) real maps for half a subperiod in the secular basis.

    populations: dense linear map applied to the density diagonal.
    coherences: elementwise decay map; its diagonal is ignored.
    """

    populations: NDArray[np.float64]
    coherences: NDArray[np.float64]


@dataclass(frozen=True)
class PeriodEntry:
    """One carrier/band channel shared by the two shaped period families.

    fallback/tail_budget: direct solver payload and allowance for ordinary tables.
    frequency: signed carrier in GHz. bound: maximum absolute gain.
    coefficients: (1,8,16,M,M) amplitude/phase polynomial for one subperiod.
    populations/coherences: half-subperiod secular maps, (M,M).
    subdivisions: positive subperiods per carrier period, at most 32.
    block_cycles: positive maximum integer-subperiod block, one when subdivided.
    project_unitary: retract interpolated U with two Newton polar iterations.
    harmonics: retained absolute Fourier cutoff, 8 for ordinary full tables.
    degrees: number of coefficients used by Clenshaw, at least one and no
        greater than coefficients.shape[1]; the highest degree is degrees - 1.
    All retained arrays are read-only. No field depends on input density.
    """

    fallback: DOP853Prepared
    tail_budget: float
    frequency: float
    bound: float
    coefficients: NDArray[np.complex128]
    populations: NDArray[np.float64]
    coherences: NDArray[np.float64]
    subdivisions: int = 1
    block_cycles: int = 1
    project_unitary: bool = False
    harmonics: int = 8
    degrees: int = 8


@dataclass(frozen=True)
class PeriodPrepared:
    """Immutable entries aligned with unique preparation keys, with owned bytes."""

    entries: tuple[PeriodEntry, ...]

    @property
    def nbytes(self) -> int:
        return sum(
            e.coefficients.nbytes
            + e.populations.nbytes
            + e.coherences.nbytes
            + e.fallback.nbytes
            for e in self.entries
        )


@njit(nogil=True, cache=True)
def _phase_coefficients(coefficients, theta, harmonics):
    result = np.zeros_like(coefficients[:, 0])
    for mode_index in range(16):
        mode = mode_index if mode_index < 8 else mode_index - 16
        if abs(mode) > harmonics:
            continue
        phase = np.exp(1j * mode * theta)
        # Preserve harmonic accumulation order without a matrix-sized temporary.
        for degree in range(coefficients.shape[0]):
            for i in range(coefficients.shape[2]):
                for j in range(coefficients.shape[3]):
                    result[degree, i, j] += (
                        phase * coefficients[degree, mode_index, i, j]
                    )
    return result


@njit(nogil=True, cache=True)
def _amplitude_unitary(coefficients, x):
    previous = np.zeros_like(coefficients[0])
    current = previous.copy()
    for degree in range(coefficients.shape[0] - 1, 0, -1):
        updated = 2 * x * current - previous + coefficients[degree]
        previous, current = current, updated
    return x * current - previous + coefficients[0]


@njit(nogil=True, cache=True)
def _phase_tables(
    coefficients, theta, subdivisions, direction, harmonics, negative_used
):
    """Contract a fixed carrier lattice; zero subdivisions skip complex playback."""
    size = coefficients.shape[-1]
    positive = np.empty(
        (subdivisions, coefficients.shape[0], size, size), dtype=np.complex128
    )
    negative = np.empty_like(positive)
    for j in range(subdivisions):
        phase = theta + 2 * np.pi * j * direction / subdivisions
        positive[j] = _phase_coefficients(coefficients, phase, harmonics)
        if negative_used:
            negative[j] = _phase_coefficients(coefficients, phase + np.pi, harmonics)
    return positive, negative


@njit(nogil=True, cache=True)
def periods(
    states: NDArray[np.complex128],
    playback: PeriodPlayback,
    unitary_table: UnitaryTable,
    dissipator: SecularMaps,
) -> None:
    """Advance states in place with immutable polynomial and secular maps.

    playback supplies aligned ports, midpoint amplitudes, start phases and
    subperiod counts. unitary_table supplies bound, coefficients, subdivisions,
    block ceiling, polar flag, carrier direction (+1/-1) and Fourier cutoff;
    dissipator supplies two half-subperiod maps.
    With block ceiling one, this is the original ordinary table playback.
    """
    ports, amplitudes, phases, cycles = playback
    (
        bound,
        coefficients,
        subdivisions,
        block_cycles,
        project_unitary,
        direction,
        harmonics,
    ) = unitary_table
    populations, coherences = dissipator
    size = states.shape[1]
    for row in range(len(states)):
        rho = states[row].copy()
        real = np.all(amplitudes[row].imag == 0)
        # Real envelopes only use two phases, including signed gains. Contract
        # Fourier modes once, rather than sixteen times per carrier cycle.
        positive, negative = _phase_tables(
            coefficients[ports[row]],
            phases[row],
            subdivisions if real else 0,
            direction,
            harmonics,
            np.any(amplitudes[row].real < 0),
        )
        cycle = 0
        while cycle < cycles[row]:
            count = min(block_cycles, cycles[row] - cycle)
            amplitude = (
                amplitudes[row, cycle + (count - 1) // 2]
                + amplitudes[row, cycle + count // 2]
            ) / 2
            x = 2 * abs(amplitude) / bound - 1
            if real:
                phase_coefficients = (
                    positive[cycle % subdivisions]
                    if amplitude.real >= 0
                    else negative[cycle % subdivisions]
                )
            else:
                phase_coefficients = _phase_coefficients(
                    coefficients[ports[row]],
                    phases[row] + np.angle(amplitude),
                    harmonics,
                )
            u = _amplitude_unitary(phase_coefficients, x)
            if project_unitary:
                # Two Newton polar steps make norm drift quadratic twice,
                # without changing or normalizing the evolving density.
                eye = np.eye(size, dtype=np.complex128)
                for _ in range(2):
                    u = u @ (3 * eye - u.conj().T @ u) / 2
            population = populations
            coherence = coherences
            if count > 1:
                population = np.linalg.matrix_power(populations, count)
                coherence = coherences**count
                u = np.linalg.matrix_power(u, count)
            for half in range(2):
                diagonal = np.diag(rho).copy()
                rho *= coherence
                for i in range(size):
                    rho[i, i] = 0j
                    for j in range(size):
                        rho[i, i] += population[i, j] * diagonal[j]
                if half == 0:
                    rho = u @ rho @ u.conj().T
            cycle += count
        states[row] = rho


def envelopes(
    table: SegmentTable, rows: NDArray[np.int64], elapsed: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Evaluate playback in a row batch using the common envelope interface."""
    values = np.empty(elapsed.shape, dtype=np.complex128)
    envelope_ids = table.tone_envelope[rows, 0]
    for index in np.unique(envelope_ids):
        envelope = table.envelopes[int(index)]
        mask = envelope_ids == index
        local = elapsed[mask] + table.tone_envelope_offset_ns[rows[mask], 0, None]
        values[mask] = envelope.values(local)
    return values * table.tone_amplitude[rows, 0, None]


def band_keys(
    table: SegmentTable, rows: NDArray[np.int64], work_plan: WorkPlan
) -> NDArray[np.void]:
    """Encode level, port, signed carrier and power-of-two peak-gain band.

    Shape, duration and phase do not enter this ordinary preparation identity.
    Zero peak gain uses the DC key, independent of its requested carrier.
    """
    peaks = envelope_bounds(table, work_plan).peaks
    maximum = (
        np.abs(table.tone_amplitude[rows, 0]) * peaks[table.tone_envelope[rows, 0]]
    )
    band = np.ceil(np.log2(np.where(maximum > 0, maximum, 1))).astype(np.int64)
    keys = np.zeros(
        len(rows),
        dtype=[
            ("level", "i8", (table.level.shape[1],)),
            ("port", "i8"),
            ("frequency", "f8"),
            ("band", "i8"),
        ],
    )
    keys["level"] = table.level[rows]
    keys["port"] = table.tone_port[rows, 0]
    keys["frequency"] = np.where(maximum > 0, table.tone_frequency_ghz[rows, 0], 0)
    keys["band"] = band
    return keys


def advance_tail(
    entry: PeriodEntry,
    table: SegmentTable,
    rows: NDArray[np.int64],
    starts: NDArray[np.float64],
    states: NDArray[np.complex128],
) -> None:
    """Advance residual intervals with the existing compiled strict DOP853.

    Shift carrier phase and envelope-local time together. Tail rows are
    independent even when the caller supplies touching same-timeline segments.
    """
    durations = table.duration_ns[rows] - starts
    selected = np.flatnonzero(durations > 0)
    if not len(selected):
        return
    active = rows[selected]
    elapsed = starts[selected, None]
    local_rows = np.arange(len(active), dtype=np.int64)
    tail_table = SegmentTable(
        timeline=local_rows,
        start_ns=np.zeros(len(active)),
        duration_ns=durations[selected],
        level=table.level[active],
        tones=table.tones[active],
        tone_port=table.tone_port[active],
        tone_frequency_ghz=table.tone_frequency_ghz[active],
        tone_amplitude=table.tone_amplitude[active],
        tone_theta_rad=table.tone_theta_rad[active]
        + 2 * np.pi * table.tone_frequency_ghz[active] * elapsed,
        tone_envelope=table.tone_envelope[active],
        tone_envelope_offset_ns=table.tone_envelope_offset_ns[active] + elapsed,
        envelopes=table.envelopes,
    )
    chunk = states[selected].copy()
    frame = np.where(
        (tail_table.tones == 1) & (tail_table.tone_amplitude[:, 0] != 0),
        "interaction",
        "lab",
    )
    plan = DOP853Plan(frame, np.full(len(active), -1, dtype=np.int64))
    advance_endpoints(
        entry.fallback.models,
        pack_segments(tail_table),
        (local_rows, np.zeros(len(active), dtype=np.int64)),
        chunk,
        entry.tail_budget,
        plan,
    )
    states[selected] = chunk
