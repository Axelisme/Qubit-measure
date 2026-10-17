"""Midpoint-frozen carrier periods, with Chebyshev/Fourier unitary tables."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource, SecularDrive, SecularModel
from ._shaped import drive_bounds, envelope_bounds
from .dop853 import DOP853Kernel


@dataclass(frozen=True)
class _Entry:
    fallback: Prepared
    tail_budget: float
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
            e.coefficients.nbytes
            + e.populations.nbytes
            + e.coherences.nbytes
            + e.fallback.nbytes
            for e in self.entries
        )


def _unitary_table(
    model: SecularModel, port: int, frequency: float, bound: float
) -> NDArray[np.complex128]:
    """Return amplitude/phase coefficients for the row's actual drive port."""
    size = len(model.energies_ghz)
    angles = np.pi * (np.arange(8) + 0.5) / 8
    amplitudes = bound * (1 + np.cos(angles)) / 2
    phases = 2 * np.pi * np.arange(16) / 16
    transform = 2 / 8 * np.cos(np.arange(8)[:, None] * angles)
    transform[0] /= 2
    tables = []
    # A common energy shift is unobservable and reduces low-carrier ODE work.
    energies = model.energies_ghz
    h0 = np.diag(energies - (energies[0] + energies[-1]) / 2)
    for drive in (model.drives[port],):

        def rhs(
            time: float,
            flat: NDArray[np.complex128],
            drive: SecularDrive = drive,
        ) -> NDArray[np.complex128]:
            z = amplitudes[:, None] * np.exp(
                1j * (phases[None, :] + 2 * np.pi * frequency * time)
            )
            h = (
                h0
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
def _phase_coefficients(coefficients, theta):
    result = np.zeros_like(coefficients[:, 0])
    for mode_index in range(16):
        mode = mode_index if mode_index < 8 else mode_index - 16
        result += np.exp(1j * mode * theta) * coefficients[:, mode_index]
    return result


@njit(nogil=True, cache=True)
def _amplitude_unitary(coefficients, x):
    previous = np.zeros_like(coefficients[0])
    current = previous.copy()
    for degree in range(7, 0, -1):
        updated = 2 * x * current - previous + coefficients[degree]
        previous, current = current, updated
    return x * current - previous + coefficients[0]


@njit(nogil=True, cache=True)
def _periods(states, playback, unitary_table, dissipator):
    """Apply independent rows in one single-threaded, GIL-free batch."""
    ports, amplitudes, phases, cycles = playback
    bound, coefficients = unitary_table
    populations, coherences = dissipator
    size = states.shape[1]
    for row in range(len(states)):
        rho = states[row].copy()
        real = np.all(amplitudes[row].imag == 0)
        # Real envelopes only use two phases, including signed gains. Contract
        # Fourier modes once, rather than sixteen times per carrier cycle.
        positive = _phase_coefficients(coefficients[ports[row]], phases[row])
        negative = _phase_coefficients(coefficients[ports[row]], phases[row] + np.pi)
        for cycle in range(cycles[row]):
            amplitude = amplitudes[row, cycle]
            x = 2 * abs(amplitude) / bound - 1
            if real:
                phase_coefficients = positive if amplitude.real >= 0 else negative
            else:
                phase_coefficients = _phase_coefficients(
                    coefficients[ports[row]], phases[row] + np.angle(amplitude)
                )
            u = _amplitude_unitary(phase_coefficients, x)
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
    DOP853Kernel().apply(
        entry.fallback,
        np.zeros(len(active), dtype=np.int64),
        tail_table,
        local_rows,
        chunk,
        entry.tail_budget,
    )
    states[selected] = chunk


class PeriodTableKernel:
    """Shaped one-tone kernel; the engine owns amortized preparation dispatch.

    Keys contain flux level, port, signed carrier and a power-of-two amplitude
    band. The engine namespaces keys by ModelSource identity. Phase, gain,
    envelope and duration can vary at apply. Eight Chebyshev nodes cover [0, 2**band]; sixteen Fourier samples
    cover phase. Zero drives use the DC direct-integration path. Model-only
    guards limit envelope motion, table bandwidth and secular splitting; their
    calibrated allowance is 1e-2 rather than the general per-segment budget.
    """

    name = "period_table"

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Check model-only envelope, interpolation and splitting estimates.

        Rejected rows need direct integration. These calibrated estimates do
        not certify arbitrary multilevel dynamics against the segment budget.
        No density is read and no trial evolution is performed.
        """
        del budget
        keys = self.prepare_keys(table, rows)
        frequencies = np.abs(keys["frequency"])
        period = 1 / np.where(frequencies > 0, frequencies, 1)
        strength, gamma = drive_bounds(table, rows, source)
        peaks, slopes = envelope_bounds(table)
        envelopes = table.tone_envelope[rows, 0]
        gain = np.abs(table.tone_amplitude[rows, 0])
        motion = 2 * np.pi * gain * strength * slopes[envelopes] * period**2
        bandwidth = np.exp2(keys["band"]) * strength * period
        split = (
            table.duration_ns[rows]
            * period**2
            * gamma
            * (gain * peaks[envelopes] * strength) ** 2
        )
        # DC and zero gain use direct integration in apply.
        return (frequencies == 0) | (
            (motion <= 0.05) & (bandwidth <= 0.5) & (split <= 0.01)
        )

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Share a port's band table across shape, duration and phase changes."""
        peaks, _ = envelope_bounds(table)
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
        tail_budget = min(budget, 1e-8)
        fallback = DOP853Kernel()
        entries = []
        for index, key in enumerate(keys):
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
                coefficients = _unitary_table(model, int(key["port"]), frequency, bound)
            for array in (coefficients, populations, coherences):
                array.setflags(write=False)
            representative = rows[index : index + 1]
            tail_data = fallback.prepare(
                fallback.prepare_keys(table, representative),
                table,
                representative,
                source,
                tail_budget,
            )
            entries.append(
                _Entry(
                    tail_data,
                    tail_budget,
                    frequency,
                    bound,
                    coefficients,
                    populations,
                    coherences,
                )
            )
        return _Prepared(tuple(entries))

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> None:
        """Decline guard preparation handoff; the engine calls guard normally."""
        del table, rows, source, budget

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        budget: float,
    ) -> None:
        """Decline input binding; apply uses the model-only prepared payload."""
        del prepared, key_index, table, rows, source_initial, budget

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
                        np.zeros(len(selected), dtype=np.int64),
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
