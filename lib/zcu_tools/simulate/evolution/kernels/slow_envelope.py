"""Midpoint-held full Hamiltonians with secular Strang bath half-steps."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import ApplyContext, KernelContext, Prepared, SegmentTable
from ._secular import coherence_decay, population_generator


@dataclass(frozen=True)
class _Channel:
    unitaries: NDArray[np.complex128]
    population_half: NDArray[np.float64]
    coherence_half: NDArray[np.float64]

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (self.unitaries, self.population_half, self.coherence_half)
        )


@dataclass(frozen=True)
class _Prepared:
    channels: tuple[_Channel, ...]

    @property
    def nbytes(self) -> int:
        return sum(channel.nbytes for channel in self.channels)


def _bath(state: NDArray[np.complex128], channel: _Channel) -> NDArray[np.complex128]:
    diagonal = channel.population_half @ np.diag(state)
    result = state * channel.coherence_half
    np.fill_diagonal(result, diagonal)
    return result


class SlowEnvelopeKernel:
    """Single constant-tone full-Hamiltonian propagation, without RWA.

    Each carrier period has source.slow_envelope_segments_per_period midpoint
    holds (default 20). A row uses max(1, ceil(duration/(T/segments))) equal
    steps, where T=1/abs(f), with a secular bath half-step around each unitary. DC uses
    one step. Time is ns, carrier is signed GHz and theta is the phase at row
    start. Prepare keys include phase, gain, duration, level, shift, port and
    segment policy; the engine namespaces them by kernel and source identity.
    Prepared arrays contain no density and are immutable. This calibrated
    approximation does not adapt to the requested budget or certify error for
    arbitrary models, pulse strengths or coherences. The guard admits constant
    tones through 30 MHz; production dispatch uses it strictly below 30 MHz.
    """

    name = "slow_envelope"
    guard_builds = False

    def guard(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.bool_]:
        """Accept one constant tone with |carrier| at most the Floquet threshold."""
        del context
        from .floquet_period import MIN_FREQUENCY_GHZ

        if not table.tone_frequency_ghz.shape[1]:
            return np.zeros(len(rows), dtype=np.bool_)
        return (
            (table.tones[rows] == 1)
            & table.constant_envelopes(rows)
            & (np.abs(table.tone_frequency_ghz[rows, 0]) <= MIN_FREQUENCY_GHZ)
        )

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> NDArray[np.void]:
        """Encode every model/channel input, including the subdivision policy."""
        keys = np.empty(
            len(rows),
            dtype=[
                ("level", "<i8", (table.level.shape[1],)),
                ("shift", "S64"),
                ("port", "<i8"),
                ("frequency", "<f8"),
                ("amplitude", "<f8"),
                ("duration", "<f8"),
                ("theta", "<f8"),
                ("segments", "<i8"),
            ],
        )
        keys["level"] = table.level[rows]
        keys["shift"] = table.shift_content[rows]
        keys["port"] = table.tone_port[rows, 0]
        keys["frequency"] = table.tone_frequency_ghz[rows, 0]
        keys["amplitude"] = table.tone_amplitude[rows, 0]
        keys["duration"] = table.duration_ns[rows]
        keys["theta"] = table.tone_theta_rad[rows, 0]
        keys["segments"] = context.source.slow_envelope_segments_per_period
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate relative work by carrier cycles, with a one-step minimum."""
        return np.maximum(
            1.0, table.duration_ns[rows] * np.abs(table.tone_frequency_ghz[rows, 0])
        )

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        context: KernelContext,
    ) -> Prepared:
        """Build full-Hamiltonian unitaries and exact secular bath half-maps."""
        del table, rows
        channels = []
        for key in keys:
            model = context.source.shifted_model(
                tuple(int(v) for v in key["level"]), bytes(key["shift"])
            )
            duration, frequency = float(key["duration"]), float(key["frequency"])
            target_dt = (
                (1.0 / abs(frequency)) / int(key["segments"]) if frequency else np.inf
            )
            steps = max(1, int(np.ceil(duration / target_dt)))
            dt = duration / steps
            drive = model.drives[int(key["port"])]
            x = float(key["amplitude"]) * drive.in_phase_ghz
            y = float(key["amplitude"]) * drive.quadrature_ghz
            h0 = np.diag(model.energies_ghz).astype(np.complex128)
            unitaries = []
            for step in range(steps):
                theta = float(key["theta"]) + 2 * np.pi * frequency * dt * (step + 0.5)
                h = h0 + np.cos(theta) * x + np.sin(theta) * y
                unitaries.append(expm(-2j * np.pi * dt * h))
            u = np.asarray(unitaries, dtype=np.complex128)
            p = np.asarray(
                expm(population_generator(model.rates_per_ns) * dt / 2),
                dtype=np.float64,
            )
            c = np.exp(-coherence_decay(model) * dt / 2)
            for array in (u, p, c):
                array.setflags(write=False)
            channels.append(_Channel(u, p, c))
        return _Prepared(tuple(channels))

    def guard_prepared(
        self, table: SegmentTable, rows: NDArray[np.int64], context: KernelContext
    ) -> None:
        """Build no guard payload; ordinary preparation owns the channels."""
        del table, rows, context

    def bind_input(
        self,
        prepared: Prepared,
        key_index: int,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source_initial: NDArray[np.complex128],
        context: KernelContext,
    ) -> None:
        """Keep model-only channels, without binding an input density."""
        del prepared, key_index, table, rows, source_initial, context

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        context: ApplyContext,
    ) -> NDArray[np.bool_]:
        """Apply D/2-U-D/2 in place; return False without an error certificate."""
        context.require_no_plan()
        del table
        if not isinstance(prepared, _Prepared):
            raise ValueError("slow_envelope requires its own prepared payload")
        for index, key in enumerate(key_index):
            channel = prepared.channels[int(key)]
            state = states[index]
            for unitary in channel.unitaries:
                state = _bath(state, channel)
                state = unitary @ state @ unitary.conj().T
                state = _bath(state, channel)
            states[index] = state
        return np.zeros(len(rows), dtype=np.bool_)
