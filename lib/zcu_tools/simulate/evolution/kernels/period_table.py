"""Midpoint-frozen carrier periods, with Chebyshev/Fourier unitary tables."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource, SecularDrive, SecularModel


@dataclass(frozen=True)
class _Entry:
    model: SecularModel
    frequency: float
    bound: float
    coefficients: NDArray[np.complex128]
    populations: NDArray[np.float64]
    coherences: NDArray[np.float64]


@dataclass(frozen=True)
class _Prepared:
    entries: tuple[_Entry, ...]

    @property
    def nbytes(self) -> int:
        return sum(
            e.coefficients.nbytes + e.populations.nbytes + e.coherences.nbytes
            for e in self.entries
        )


def _unitary_table(
    model: SecularModel, frequency: float, bound: float
) -> NDArray[np.complex128]:
    """Return Chebyshev coefficients of Fourier coefficients for every port."""
    size = len(model.energies_ghz)
    angles = np.pi * (np.arange(8) + 0.5) / 8
    amplitudes = bound * (1 + np.cos(angles)) / 2
    phases = 2 * np.pi * np.arange(16) / 16
    transform = 2 / 8 * np.cos(np.arange(8)[:, None] * angles)
    transform[0] /= 2
    tables = []
    for drive in model.drives:

        def rhs(
            time: float,
            flat: NDArray[np.complex128],
            drive: SecularDrive = drive,
        ) -> NDArray[np.complex128]:
            z = amplitudes[:, None] * np.exp(
                1j * (phases[None, :] + 2 * np.pi * frequency * time)
            )
            h = (
                np.diag(model.energies_ghz)
                + z.real[..., None, None] * drive.in_phase_ghz
                + z.imag[..., None, None] * drive.quadrature_ghz
            )
            return (-2j * np.pi * (h @ flat.reshape(8, 16, size, size))).ravel()

        solution = solve_ivp(
            rhs,
            (0, 1 / abs(frequency)),
            np.broadcast_to(np.eye(size, dtype=complex), (8, 16, size, size))
            .copy()
            .ravel(),
            method="DOP853",
            rtol=2e-11,
            atol=2e-13,
            max_step=1 / (32 * abs(frequency)),
            t_eval=[1 / abs(frequency)],
        )
        if not solution.success:
            raise RuntimeError(f"period table preparation: {solution.message}")
        values = solution.y[:, -1].reshape(8, 16, size, size)
        tables.append(
            np.einsum("ab,bkij->akij", transform, np.fft.fft(values, axis=1) / 16)
        )
    return np.asarray(tables, dtype=np.complex128)


@njit(nogil=True, cache=True)
def _periods(states, playback, unitary_table, dissipator):
    """Apply independent rows in one single-threaded, GIL-free batch."""
    ports, amplitudes, phases, cycles = playback
    bound, coefficients = unitary_table
    populations, coherences = dissipator
    size = states.shape[1]
    for row in range(len(states)):
        rho = states[row].copy()
        for cycle in range(cycles[row]):
            amplitude = amplitudes[row, cycle]
            x = 2 * abs(amplitude) / bound - 1
            theta = phases[row] + np.angle(amplitude)
            u = np.zeros((size, size), np.complex128)
            for mode_index in range(16):
                mode = mode_index if mode_index < 8 else mode_index - 16
                b1 = np.zeros((size, size), np.complex128)
                b2 = b1.copy()
                for degree in range(7, 0, -1):
                    b0 = 2 * x * b1 - b2 + coefficients[ports[row], degree, mode_index]
                    b2 = b1
                    b1 = b0
                u += np.exp(1j * mode * theta) * (
                    x * b1 - b2 + coefficients[ports[row], 0, mode_index]
                )
            for half in range(2):
                diagonal = np.diag(rho).copy()
                rho *= coherences
                for i in range(size):
                    rho[i, i] = 0j
                    for j in range(size):
                        rho[i, i] += populations[i, j] * diagonal[j]
                if half == 0:
                    rho = u @ rho @ u.conj().T
        states[row] = rho


def _envelopes(
    table: SegmentTable, rows: NDArray[np.int64], elapsed: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Evaluate playback in a row batch using the common envelope interface."""
    values = np.empty(elapsed.shape, dtype=np.complex128)
    for index, envelope in enumerate(table.envelopes):
        mask = table.tone_envelope[rows, 0] == index
        local = elapsed[mask] + table.tone_envelope_offset_ns[rows[mask], 0, None]
        values[mask] = envelope.values(local)
    return values * table.tone_amplitude[rows, 0, None]


def _tail(
    entry: _Entry,
    table: SegmentTable,
    rows: NDArray[np.int64],
    starts: NDArray[np.float64],
    states: NDArray[np.complex128],
) -> None:
    """Integrate all residual intervals jointly, with full secular dissipation."""
    durations = table.duration_ns[rows] - starts
    if not np.any(durations > 0):
        return
    model = entry.model
    size = states.shape[-1]
    ports = table.tone_port[rows, 0]
    x = np.stack([d.in_phase_ghz for d in model.drives])[ports]
    y = np.stack([d.quadrature_ghz for d in model.drives])[ports]
    outgoing = model.rates_per_ns.sum(axis=0)
    decay = (outgoing[:, None] + outgoing[None, :]) / 2 + model.dephasing_per_ns
    diagonal = np.arange(size)

    def rhs(time: float, flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        elapsed = starts + durations * time
        amplitude = _envelopes(table, rows, elapsed[:, None])[:, 0]
        z = amplitude * np.exp(
            1j * (table.tone_theta_rad[rows, 0] + 2 * np.pi * entry.frequency * elapsed)
        )
        h = (
            np.diag(model.energies_ghz)
            + z.real[:, None, None] * x
            + z.imag[:, None, None] * y
        )
        rho = flat.reshape(states.shape)
        derivative = -2j * np.pi * (h @ rho - rho @ h) - decay * rho
        derivative[:, diagonal, diagonal] += (
            np.diagonal(rho, axis1=1, axis2=2) @ model.rates_per_ns.T
        )
        return (durations[:, None, None] * derivative).ravel()

    solution = solve_ivp(
        rhs,
        (0, 1),
        states.ravel().copy(),
        method="DOP853",
        rtol=2e-10,
        atol=2e-12,
        max_step=min(1.0, 0.05 / float(durations.max())),
        t_eval=[1.0],
    )
    if not solution.success:
        raise RuntimeError(f"period table tail: {solution.message}")
    states[:] = solution.y[:, -1].reshape(states.shape)


class PeriodTableKernel:
    """Shaped one-tone kernel; the engine owns shape and reuse-count dispatch.

    Keys contain flux level, signed carrier and a power-of-two amplitude band.
    The engine namespaces keys by ModelSource identity. Tables include every
    model port, allowing port, phase, gain, envelope and duration to vary at
    apply. Eight Chebyshev nodes cover [0, 2**band]; sixteen Fourier samples
    cover phase. Zero drives use the DC direct-integration path. No numerical
    validity guard is imposed in this first version; its accuracy allowance is
    1e-2 rather than the general per-segment budget.
    """

    name = "period_table"

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Accept dispatched shaped single-tone rows without an error guard."""
        del table, source, budget
        return np.ones(len(rows), dtype=np.bool_)

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Share a band table across shape, duration, phase and port changes."""
        peaks = np.ones(len(table.envelopes))
        for index, envelope in enumerate(table.envelopes):
            if envelope.kind == "sampled":
                peaks[index] = np.max(np.abs(envelope.samples))
        maximum = (
            np.abs(table.tone_amplitude[rows, 0]) * peaks[table.tone_envelope[rows, 0]]
        )
        band = np.ceil(np.log2(np.where(maximum > 0, maximum, 1))).astype(np.int64)
        keys = np.zeros(
            len(rows),
            dtype=[
                ("level", "i8", (table.level.shape[1],)),
                ("frequency", "f8"),
                ("band", "i8"),
            ],
        )
        keys["level"] = table.level[rows]
        keys["frequency"] = np.where(maximum > 0, table.tone_frequency_ghz[rows, 0], 0)
        keys["band"] = band
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate period count, using direct-solver steps for DC rows."""
        frequencies = np.abs(table.tone_frequency_ghz[rows, 0])
        return 1 + table.duration_ns[rows] * np.where(frequencies > 0, frequencies, 20)

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> Prepared:
        """Build immutable band tables and half-period dissipators, without rho.

        rows are representative table rows aligned with keys; numerical solver
        failure raises RuntimeError. DC entries retain only model data.
        """
        del table, rows, budget
        entries = []
        for key in keys:
            model = source.model(tuple(int(v) for v in key["level"]))
            frequency = float(key["frequency"])
            bound = float(2.0 ** int(key["band"]))
            size = len(model.energies_ghz)
            populations = np.eye(size)
            coherences = np.ones((size, size))
            coefficients = np.empty((0, 8, 16, size, size), dtype=np.complex128)
            if frequency != 0:
                half = 0.5 / abs(frequency)
                outgoing = model.rates_per_ns.sum(axis=0)
                populations = np.asarray(
                    expm(half * (model.rates_per_ns - np.diag(outgoing))),
                    dtype=np.float64,
                )
                coherences = np.exp(
                    -half
                    * (
                        (outgoing[:, None] + outgoing[None, :]) / 2
                        + model.dephasing_per_ns
                    )
                )
                coefficients = _unitary_table(model, frequency, bound)
            for array in (coefficients, populations, coherences):
                array.setflags(write=False)
            entries.append(
                _Entry(model, frequency, bound, coefficients, populations, coherences)
            )
        return _Prepared(tuple(entries))

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> NDArray[np.bool_]:
        """Advance rows in place with per-period Strang and directly solved tails.

        Negative carriers use period 1/abs(f); DC integrates the whole shape.
        A payload from another kernel raises ValueError. Nonfinite output or
        integration failure raises RuntimeError, never a silent fallback.

        Return an all-False bool mask aligned with rows; no post-apply check.
        """
        del budget
        if not isinstance(prepared, _Prepared):
            raise ValueError("period_table requires its own prepared payload")
        for index, entry in enumerate(prepared.entries):
            mask = key_index == index
            if not np.any(mask):
                continue
            selected = rows[mask]
            chunk = states[mask].copy()
            starts = np.zeros(len(selected))
            if entry.frequency != 0:
                period = 1 / abs(entry.frequency)
                cycles = np.floor(table.duration_ns[selected] / period).astype(np.int64)
                elapsed = (np.arange(int(cycles.max()))[None, :] + 0.5) * period
                elapsed = np.asarray(
                    np.broadcast_to(elapsed, (len(selected), elapsed.shape[1])),
                    dtype=np.float64,
                )
                amplitudes = _envelopes(table, selected, elapsed)
                _periods(
                    chunk,
                    (
                        table.tone_port[selected, 0],
                        amplitudes,
                        table.tone_theta_rad[selected, 0],
                        cycles,
                    ),
                    (entry.bound, entry.coefficients),
                    (entry.populations, entry.coherences),
                )
                starts = np.asarray(cycles * period, dtype=np.float64)
            _tail(entry, table, selected, starts, chunk)
            if not np.isfinite(chunk).all():
                raise RuntimeError(f"period table nonfinite density in rows {selected}")
            states[mask] = chunk
        return np.zeros(len(rows), dtype=np.bool_)
