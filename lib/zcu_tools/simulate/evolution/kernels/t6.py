"""Constant-tone secular channels, powered by periods and interpolated in phase."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource, SecularModel


@dataclass(frozen=True)
class _ChannelProblem:
    static: NDArray[np.complex128]
    cosine: NDArray[np.complex128]
    sine: NDArray[np.complex128]
    frequency: float
    duration: float
    steps: int
    tolerance: float

    def channel(self, theta: float) -> NDArray[np.complex128]:
        """Build one full-duration channel, with no input density involved."""
        size = len(self.static)
        identity = np.eye(size, dtype=np.complex128)
        if self.duration == 0:
            return identity
        if self.frequency == 0:
            generator = (
                self.static + np.cos(theta) * self.cosine + np.sin(theta) * self.sine
            )
            result = expm(self.duration * generator)
        else:
            period = 1 / abs(self.frequency)
            count = int(self.duration // period)
            tail = self.duration - count * period
            whole = self.integrate(period, theta) if count else identity
            result = np.linalg.matrix_power(whole, count)
            if tail:
                result = self.integrate(tail, theta) @ result
        if not np.isfinite(result).all():
            raise RuntimeError("t6: nonfinite periodic channel")
        return np.asarray(result, dtype=np.complex128)

    def integrate(self, duration: float, theta: float) -> NDArray[np.complex128]:
        """Integrate a period or its tail in r-selected physical substeps."""
        size = len(self.static)
        channel = np.eye(size, dtype=np.complex128)
        period = 1 / abs(self.frequency)
        boundaries = np.linspace(
            0, duration, max(1, int(np.ceil(duration / period * self.steps))) + 1
        )

        def rhs(time: float, flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
            phase = theta + 2 * np.pi * self.frequency * time
            generator = (
                self.static + np.cos(phase) * self.cosine + np.sin(phase) * self.sine
            )
            return (generator @ flat.reshape(size, size)).ravel()

        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            solution = solve_ivp(
                rhs,
                (start, stop),
                channel.ravel(),
                method="DOP853",
                rtol=self.tolerance,
                atol=self.tolerance * 0.01,
                max_step=period / self.steps,
            )
            if not solution.success:
                raise RuntimeError(
                    f"t6: periodic integration failed: {solution.message}"
                )
            channel = solution.y[:, -1].reshape(size, size)
        return channel


def _commutator(hamiltonian: NDArray[np.complex128]) -> NDArray[np.complex128]:
    identity = np.eye(len(hamiltonian))
    return (
        -2j
        * np.pi
        * (np.kron(hamiltonian, identity) - np.kron(identity, hamiltonian.T))
    )


def _problem(
    model: SecularModel, table: SegmentTable, row: int, budget: float
) -> _ChannelProblem:
    drive = model.drives[table.tone_port[row, 0]]
    amplitude = float(table.tone_amplitude[row, 0])
    frequency = float(table.tone_frequency_ghz[row, 0])
    duration = float(table.duration_ns[row])
    energies = model.energies_ghz
    outgoing = model.rates_per_ns.sum(axis=0)
    decay = (outgoing[:, None] + outgoing[None, :]) / 2 + model.dephasing_per_ns
    static = _commutator(np.diag(energies).astype(np.complex128)) - np.diag(
        decay.ravel()
    )
    populations = np.arange(len(energies)) * (len(energies) + 1)
    static[np.ix_(populations, populations)] = model.rates_per_ns - np.diag(outgoing)
    coupling = 2 * np.abs(amplitude * drive.in_phase_ghz)
    detuning = np.abs(energies[:, None] - energies[None, :] - frequency)
    driven = (coupling > 0) & ~np.eye(len(energies), dtype=bool)
    ratios = np.divide(
        coupling, detuning, out=np.full_like(coupling, np.inf), where=detuning != 0
    )
    ratio = float(np.max(ratios[driven], initial=0))
    steps = 1 if ratio <= 0.1 else 4 if ratio <= 0.2 else 8
    # Powering repeats period error, so tighten the solve before taking powers.
    tolerance = max(
        2.3e-14, min(1e-8, budget * 0.002 / max(1, duration * abs(frequency)))
    )
    return _ChannelProblem(
        static,
        _commutator(amplitude * drive.in_phase_ghz),
        _commutator(amplitude * drive.quadrature_ghz),
        frequency,
        duration,
        steps,
        tolerance,
    )


def _fourier(problem: _ChannelProblem, budget: float) -> NDArray[np.complex128]:
    """Refine phase sampling until off-grid channels and spectral tails agree."""
    probes = np.array([0.371, 2.137, 5.319])
    exact = np.stack([problem.channel(float(theta)) for theta in probes])
    for count in (16, 32, 64, 128, 256, 512):
        values = np.stack(
            [problem.channel(2 * np.pi * k / count) for k in range(count)]
        )
        coefficients = np.fft.fft(values, axis=0) / count
        modes = np.fft.fftfreq(count, 1 / count)
        interpolated = np.einsum(
            "pk,kij->pij", np.exp(1j * probes[:, None] * modes), coefficients
        )
        error = np.max(np.sum(np.abs(interpolated - exact), axis=-1))
        # The upper spectral shell must also be small; isolated probe agreement
        # alone could hide aliasing between the off-grid validation phases.
        shell = np.abs(modes) >= count / 4
        tail = np.max(np.sum(np.abs(coefficients[shell]), axis=(0, 2)))
        if error <= budget / 16 and tail <= budget / 16:
            return coefficients
    raise RuntimeError("t6: phase Fourier interpolation did not meet budget")


@dataclass(frozen=True)
class _T6Prepared:
    coefficients: NDArray[np.complex128]
    modes: NDArray[np.float64]
    counts: NDArray[np.int64]

    @property
    def nbytes(self) -> int:
        return self.coefficients.nbytes + self.modes.nbytes + self.counts.nbytes


@njit(nogil=True)
def _apply_channels(coefficients, modes, counts, key_index, theta, states):
    size = states.shape[1]
    dimension = size * size
    for row in range(len(states)):
        key = key_index[row]
        channel = np.zeros((dimension, dimension), dtype=np.complex128)
        for k in range(counts[key]):
            channel += np.exp(1j * modes[key, k] * theta[row]) * coefficients[key, k]
        states[row] = (channel @ states[row].reshape(dimension)).reshape(size, size)


class T6Kernel:
    """One constant tone at any strength, with full secular dissipation.

    Prepare keys contain flux level, port, signed carrier GHz, signed gain and
    duration ns. The engine namespaces keys by the DC model source identity.
    Phase and start time are apply-only. Prepare builds density-independent
    Fourier channels; apply only evaluates those channels and multiplies states.
    A positive finite budget is required; integration or interpolation failure
    raises RuntimeError rather than silently selecting another algorithm.
    """

    name = "t6"

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Accept exactly one unity envelope, irrespective of branch priority."""
        del source
        _check_budget(budget)
        accepted = table.tones[rows] == 1
        if table.tone_envelope.shape[1]:
            constant = np.array(
                [envelope.kind == "constant" for envelope in table.envelopes]
            )
            active = np.flatnonzero(accepted)
            accepted[active] &= constant[table.tone_envelope[rows[active], 0]]
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Return fixed-width keys; equal rows share a phase-independent channel."""
        keys = np.empty(
            len(rows),
            dtype=[
                ("level", "<i8", (table.level.shape[1],)),
                ("port", "<i8"),
                ("frequency", "<f8"),
                ("amplitude", "<f8"),
                ("duration", "<f8"),
            ],
        )
        keys["level"] = table.level[rows]
        keys["port"] = table.tone_port[rows, 0]
        keys["frequency"] = table.tone_frequency_ghz[rows, 0]
        keys["amplitude"] = table.tone_amplitude[rows, 0]
        keys["duration"] = table.duration_ns[rows]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate channel evaluation cost, with a small powering allowance."""
        return 1 + np.log1p(
            table.duration_ns[rows] * np.abs(table.tone_frequency_ghz[rows, 0])
        )

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> Prepared:
        """Build immutable Fourier channels from representatives aligned to keys."""
        _check_budget(budget)
        if len(keys) != len(rows) or not np.array_equal(
            keys, self.prepare_keys(table, rows)
        ):
            raise ValueError("t6: keys and representative rows must align")
        if not self.guard(table, rows, source, budget).all():
            raise ValueError("t6: prepare requires one constant tone")
        entries = [
            _fourier(
                _problem(
                    source.model(tuple(int(v) for v in table.level[row])),
                    table,
                    int(row),
                    budget,
                ),
                budget,
            )
            for row in rows
        ]
        return _pack(entries)

    def apply(
        self,
        prepared: Prepared,
        key_index: NDArray[np.int64],
        table: SegmentTable,
        rows: NDArray[np.int64],
        states: NDArray[np.complex128],
        budget: float,
    ) -> None:
        """Evaluate phase Fourier series and advance the aligned densities in place."""
        _check_budget(budget)
        if not isinstance(prepared, _T6Prepared):
            raise ValueError("t6: incompatible prepared payload")
        _apply_channels(
            prepared.coefficients,
            prepared.modes,
            prepared.counts,
            key_index,
            table.tone_theta_rad[rows, 0],
            states,
        )
        if not np.isfinite(states).all():
            raise RuntimeError("t6: nonfinite density after channel application")


def _pack(entries: list[NDArray[np.complex128]]) -> _T6Prepared:
    count = max((len(entry) for entry in entries), default=0)
    size = entries[0].shape[1] if entries else 0
    coefficients = np.zeros((len(entries), count, size, size), dtype=np.complex128)
    modes = np.zeros((len(entries), count))
    counts = np.array([len(entry) for entry in entries], dtype=np.int64)
    for i, entry in enumerate(entries):
        coefficients[i, : len(entry)] = entry
        modes[i, : len(entry)] = np.fft.fftfreq(len(entry), 1 / len(entry))
    for array in (coefficients, modes, counts):
        array.flags.writeable = False
    return _T6Prepared(coefficients, modes, counts)


def _check_budget(budget: float) -> None:
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("t6: budget must be positive and finite")
