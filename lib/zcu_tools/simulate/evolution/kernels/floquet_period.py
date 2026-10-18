"""Constant-tone Hilbert propagators with compact secular Strang stepping."""

from copy import copy
from dataclasses import dataclass
from threading import RLock
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..declarations import (
    AlgorithmIdentity,
    Eligibility,
    GuardDeclaration,
    KernelDeclaration,
    ReuseDeclaration,
)
from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource, SecularModel
from ._dop853 import Tolerances, integrate_dense

_MIN_FREQUENCY_GHZ = 0.03
_MAX_SPLIT_ESTIMATE = 1.0
_MAX_SUBDIVISIONS = 64
_MAGNUS_FREQUENCY_GHZ = 1.0
_GAIN_BLOCK_SIZE = 32
_GAIN_MAX_STRIPES = 12


class _DensePeriod(NamedTuple):
    # Numeric-only view for compiled high-carrier endpoint composition and steps.
    times: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    unitary: NDArray[np.complex128]
    population_half: NDArray[np.complex128]
    coherence_half: NDArray[np.float64]


@dataclass(frozen=True)
class _Period:
    # Dense DOP853 coefficients, or CF4 prefix unitaries plus Hamiltonian data.
    times: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    unitary: NDArray[np.complex128]
    generator: NDArray[np.float64]
    decay: NDArray[np.float64]
    population_half: NDArray[np.complex128]
    coherence_half: NDArray[np.float64]
    split_rate: float
    magnus: (
        tuple[
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            float,
        ]
        | None
    )

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.times,
                self.coefficients,
                self.unitary,
                self.generator,
                self.decay,
                self.population_half,
                self.coherence_half,
            )
        ) + (sum(a.nbytes for a in self.magnus[:3]) if self.magnus is not None else 0)


@dataclass(frozen=True)
class _Periods:
    entries: tuple[_Period, ...]

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries)


def _check_budget(budget: float) -> None:
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("Floquet period budget must be finite and positive")


def _bath(model: SecularModel) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    outgoing = model.rates_per_ns.sum(axis=0)
    return (
        model.rates_per_ns - np.diag(outgoing),
        (outgoing[:, None] + outgoing[None, :]) / 2 + model.dephasing_per_ns,
    )


def _split_rate(model: SecularModel, port: int, frequency: float, gain: float) -> float:
    generator, decay = _bath(model)
    gamma = max(float(np.linalg.norm(generator, 2)), float(np.max(decay)))
    drive = model.drives[port]
    strength = (
        2
        * np.pi
        * abs(gain)
        * (
            np.linalg.norm(drive.in_phase_ghz, 2)
            + np.linalg.norm(drive.quadrature_ghz, 2)
        )
    )
    return float(gamma * (strength / frequency) ** 2)


def _subdivisions(estimate: float) -> int:
    return max(1, int(np.ceil(np.sqrt(estimate / _MAX_SPLIT_ESTIMATE))))


@njit(nogil=True, cache=True)
def _rhs(time, flat, context):
    h0, x, y, frequency = context
    size = len(h0)
    phase = 2 * np.pi * frequency * time
    h = h0 + np.cos(phase) * x + np.sin(phase) * y
    return (-2j * np.pi * (h @ flat.reshape(size, size))).ravel()


@njit(nogil=True, cache=True)
def _integrate_dense(context, state, duration, controls):
    # A callback-free entry point caches the linked shared integrator across
    # processes; the RHS dispatcher itself has a process-specific cache key.
    return integrate_dense(_rhs, context, state, duration, controls)


@njit(nogil=True, cache=True)
def _unitary(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Retract a near-unitary Hilbert endpoint with one Newton polar step."""
    # DOP853's <= 1e-10 controls and CF4's Hermitian exponentials keep the
    # singular values near one. Newton squares that norm defect without an SVD.
    return 0.5 * (3 * matrix - matrix @ (matrix.conj().T @ matrix))


class _Hamiltonian(NamedTuple):
    """One key's lab-frame Hamiltonian in GHz, before multiplication by 2*pi.

    h0 is the diagonal static matrix with common energy zero removed; x and y
    are the gain-scaled Hermitian drive quadratures; frequency is signed GHz.
    """

    h0: NDArray[np.complex128]
    x: NDArray[np.complex128]
    y: NDArray[np.complex128]
    frequency: float


@dataclass(frozen=True)
class _Neighbour:
    gain: float
    period: _Period


class _GainBlock:
    """Serialize a fixed gain trajectory with four recent unclaimed memo keys.

    The block lock covers numerical construction. The separate inventory lock
    protects byte snapshots; readers never acquire a block lock. Keys no longer
    memoized replay the same trajectory from its start.
    """

    def __init__(
        self,
        model: SecularModel,
        port: int,
        frequency: float,
        budget: float,
        gains: tuple[float, ...],
        inventory_lock: RLock,
    ) -> None:
        self.model = model
        self.port = port
        self.frequency = frequency
        self.budget = budget
        self.gains = gains
        self.history: list[_Neighbour] = []
        self.memo: dict[float, _Period] = {}
        self.delivered: set[float] = set()
        self.next_index = 0
        self.lock = RLock()
        self.inventory_lock = inventory_lock
        self.attempts = 0
        self.hits = 0

    @property
    def nbytes(self) -> int:
        # Caller holds inventory_lock. Conservative if history also owns memo.
        return sum(n.period.nbytes for n in self.history) + sum(
            period.nbytes for period in self.memo.values()
        )

    def build(self, gain: float) -> _Period:
        """Return this key's deterministic payload, then release its memo entry."""
        with self.lock:
            with self.inventory_lock:
                if gain in self.memo:
                    self.delivered.add(gain)
                    return self.memo.pop(gain)
                target = self.gains.index(gain)
                if target < self.next_index:
                    self.next_index = 0
                    self.history.clear()
            while self.next_index <= target:
                coordinate = self.gains[self.next_index]
                result = _build(
                    self.model, self.port, self.frequency, coordinate, self.budget, self
                )
                with self.inventory_lock:
                    self.history = (self.history + [_Neighbour(coordinate, result)])[
                        -4:
                    ]
                    recent = {neighbour.gain for neighbour in self.history}
                    self.memo = {
                        key: period
                        for key, period in self.memo.items()
                        if key in recent
                    }
                    if coordinate not in self.delivered or coordinate == gain:
                        self.memo[coordinate] = result
                    self.next_index += 1
            with self.inventory_lock:
                self.delivered.add(gain)
                # No later key needs these prefixes once the block is built.
                if self.next_index == len(self.gains):
                    self.history.clear()
                return self.memo.pop(gain)

    def prefixes(
        self, context: _Hamiltonian, gain: float
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Check the fixed predecessors, reusing coarse work on rejection."""
        tolerance = min(1e-4, self.budget * 0.1)
        if len(self.history) >= 2:
            continued = _continued_prefixes(self.history, context, gain, tolerance)
            if continued is not None:
                times, coefficients, hit = continued
                with self.inventory_lock:
                    self.attempts += 1
                    self.hits += int(hit)
                return times, coefficients
        return _magnus_build(*context, tolerance)


class _GainWorkspace:
    """One evolve's deterministic gain blocks and numeric-byte inventory.

    Planning uses the complete table, never worker identity or chunk size.
    Runtime blocks isolate model object, port, signed carrier and budget.
    attempts/hits count checked/accepted predictors, including replayed builds.
    nbytes charges history and unclaimed memo arrays, conservatively including
    arrays also held by active payloads. No data survives the owning evolve.
    """

    def __init__(self, table: SegmentTable) -> None:
        groups: dict[tuple[tuple[int, ...], int, float], set[float]] = {}
        rows = np.flatnonzero(table.tones == 1)
        rows = rows[table.constant_envelopes(rows)]
        for row in rows:
            frequency = float(table.tone_frequency_ghz[row, 0])
            if not _MIN_FREQUENCY_GHZ <= abs(frequency) < _MAGNUS_FREQUENCY_GHZ:
                continue
            key = (
                tuple(int(v) for v in table.level[row]),
                int(table.tone_port[row, 0]),
                frequency,
            )
            groups.setdefault(key, set()).add(float(table.tone_amplitude[row, 0]))
        self._plans: dict[
            tuple[tuple[int, ...], int, float], dict[float, tuple[float, ...]]
        ] = {}
        for key, gains in groups.items():
            if len(gains) < 2:
                continue
            ordered = sorted(gains)
            # Fixed interleaved trajectories avoid holding a whole contiguous
            # scan while strided timeline chunks wait for their turn. Small
            # groups keep one trajectory so they still have useful neighbours.
            stripes = min(_GAIN_MAX_STRIPES, max(1, len(ordered) // 16))
            plan: dict[float, tuple[float, ...]] = {}
            for stripe in range(stripes):
                trajectory = ordered[stripe::stripes]
                for start in range(0, len(trajectory), _GAIN_BLOCK_SIZE):
                    block = tuple(trajectory[start : start + _GAIN_BLOCK_SIZE])
                    plan.update((gain, block) for gain in block)
            self._plans[key] = plan
        self._blocks: dict[
            tuple[int, int, float, float, tuple[float, ...]], _GainBlock
        ] = {}
        self._lock = RLock()

    @property
    def attempts(self) -> int:
        with self._lock:
            return sum(block.attempts for block in self._blocks.values())

    @property
    def hits(self) -> int:
        with self._lock:
            return sum(block.hits for block in self._blocks.values())

    @property
    def nbytes(self) -> int:
        """Current retained numeric bytes, excluding Python/model overhead."""
        with self._lock:
            return sum(block.nbytes for block in self._blocks.values())

    def build(
        self,
        model: SecularModel,
        port: int,
        frequency: float,
        gain: float,
        budget: float,
    ) -> _Period:
        """Build in fixed gain order, independent of caller or scheduling order."""
        gains = self._plans.get((model.level, port, frequency), {}).get(gain)
        if gains is None:
            return _build(model, port, frequency, gain, budget)
        identity = (id(model), port, frequency, budget, gains)
        with self._lock:
            block = self._blocks.get(identity)
            if block is None:
                block = _GainBlock(model, port, frequency, budget, gains, self._lock)
                self._blocks[identity] = block
        return block.build(gain)


def _continued_prefixes(
    neighbours: list[_Neighbour],
    context: _Hamiltonian,
    gain: float,
    tolerance: float,
) -> tuple[NDArray[np.float64], NDArray[np.complex128], bool] | None:
    # Extrapolate only gain in the interaction picture, then
    # independently check the new key with the original coarse-grid probes.
    h0, x, y, frequency = context
    count = len(neighbours[-1].period.times) - 1
    if any(
        n.period.magnus is None or len(n.period.times) != count + 1 for n in neighbours
    ):
        return None
    coordinates = [n.gain for n in neighbours]
    target = gain
    if len(set(coordinates)) != len(coordinates):
        return None
    gap = coordinates[-1] - coordinates[-2]
    if abs((target - coordinates[-1]) / gap) > 2:
        return None
    period = 1 / abs(frequency)
    width = float(
        np.max(h0.real) - np.min(h0.real) + 2 * (np.linalg.norm(x) + np.linalg.norm(y))
    )
    coarse_count = max(8, 8 * int(np.ceil(1.5 * period * width / 8)))
    if count < 8 * int(np.ceil(1.5 * coarse_count / 8)):
        return None
    times = np.linspace(0.0, period, count + 1)
    # Identical model, carrier and uniform grid give every predecessor the
    # same analytic H0 phase. The interaction-picture transform and inverse
    # cancel algebraically in this linear combination; avoid repeated complex
    # exponentials and their roundoff without changing the predictor.
    candidate = np.zeros_like(neighbours[-1].period.coefficients)
    for index, neighbour in enumerate(neighbours):
        weight = 1.0
        for other, coordinate in enumerate(coordinates):
            if other != index:
                weight *= (target - coordinate) / (coordinates[index] - coordinate)
        candidate += weight * neighbour.period.coefficients
    _, coarse = _magnus_prefixes(h0, x, y, frequency, coarse_count)
    error = float(
        np.max(
            [
                np.linalg.norm(
                    candidate[j * count // 8] - coarse[j * coarse_count // 8]
                )
                for j in range(9)
            ]
        )
    )
    if not np.isfinite(error) or error > tolerance:
        times, fine = _magnus_refine(context, tolerance, coarse_count, coarse)
        return times, fine, False
    return times, candidate, True


def _build(
    model: SecularModel,
    port: int,
    frequency: float,
    gain: float,
    budget: float,
    block: _GainBlock | None = None,
) -> _Period:
    size = len(model.energies_ghz)
    period = 1 / abs(frequency)
    drive = model.drives[port]
    # Removing the common energy zero only changes an unobservable global phase.
    h0 = np.diag(model.energies_ghz - model.energies_ghz[0]).astype(complex)
    x, y = gain * drive.in_phase_ghz, gain * drive.quadrature_ghz
    magnus = None
    if abs(frequency) < _MAGNUS_FREQUENCY_GHZ:
        # Low carriers make Schrödinger DOP853 resolve many unobservable H0 turns.
        # CF4 exponentiates H0 exactly; only carrier motion needs approximation.
        tolerance = min(1e-4, budget * 0.1)
        if block is None:
            times, coefficients = _magnus_build(h0, x, y, frequency, tolerance)
        else:
            times, coefficients = block.prefixes(
                _Hamiltonian(h0, x, y, frequency), gain
            )
        final = coefficients[-1]
        magnus = (h0, x, y, frequency)
        for array in (h0, x, y):
            array.setflags(write=False)
    else:
        tolerance = min(1e-10, budget * 1e-4)
        times, coefficients, final = _integrate_dense(
            (h0, x, y, frequency),
            np.eye(size, dtype=complex).ravel(),
            period,
            Tolerances(tolerance, tolerance * 0.01, period / 4),
        )
        coefficients = coefficients.reshape(-1, 8, size, size)
    generator, decay = _bath(model)
    arrays = (
        times,
        coefficients,
        _unitary(final.reshape(size, size)),
        generator,
        decay,
        expm(generator * period / 2).astype(complex),
        np.exp(-decay * period / 2),
    )
    for array in arrays:
        if not np.isfinite(array).all():
            raise RuntimeError("Floquet period preparation produced nonfinite data")
        array.setflags(write=False)
    return _Period(*arrays, _split_rate(model, port, frequency, gain), magnus)


@njit(nogil=True, cache=True)
def _hermitian_exponential(h, step):
    # Scaling and squaring with the [7/7] Pade approximant. Its double-
    # precision backward-error radius is 0.9504178996162932. Centering the
    # spectrum reduces scaling without changing the returned global phase.
    size = len(h)
    center = np.trace(h).real / size
    identity = np.eye(size, dtype=np.complex128)
    matrix = (-2j * np.pi * step) * (h - center * identity)
    norm = np.max(np.sum(np.abs(matrix), axis=0))
    squarings = max(0, int(np.ceil(np.log2(max(norm / 0.9504178996162932, 1.0)))))
    matrix = matrix / 2.0**squarings
    square = matrix @ matrix
    fourth = square @ square
    sixth = fourth @ square
    odd = matrix @ (8648640.0 * identity + 277200.0 * square + 1512.0 * fourth + sixth)
    even = 17297280.0 * identity + 1995840.0 * square + 25200.0 * fourth + 56.0 * sixth
    result = np.ascontiguousarray(np.linalg.solve(even - odd, even + odd))
    for _ in range(squarings):
        result = result @ result
    return np.exp(-2j * np.pi * step * center) * result


@njit(nogil=True, cache=True)
def _magnus_step(h0, x, y, frequency, time, step):
    # Fourth-order commutator-free Magnus, two Gauss nodes and two exponentials.
    a = (3 - 2 * np.sqrt(3)) / 12
    b = (3 + 2 * np.sqrt(3)) / 12
    first = 2 * np.pi * frequency * (time + (0.5 - np.sqrt(3) / 6) * step)
    last = 2 * np.pi * frequency * (time + (0.5 + np.sqrt(3) / 6) * step)
    h1 = h0 + np.cos(first) * x + np.sin(first) * y
    h2 = h0 + np.cos(last) * x + np.sin(last) * y
    return _hermitian_exponential(a * h1 + b * h2, step) @ _hermitian_exponential(
        b * h1 + a * h2, step
    )


@njit(nogil=True, cache=True)
def _magnus_prefixes(h0, x, y, frequency, count):
    period = 1 / abs(frequency)
    times = np.linspace(0.0, period, count + 1)
    prefixes = np.empty((count + 1, len(h0), len(h0)), dtype=np.complex128)
    prefixes[0] = np.eye(len(h0), dtype=np.complex128)
    for j in range(count):
        prefixes[j + 1] = (
            _magnus_step(h0, x, y, frequency, times[j], period / count) @ prefixes[j]
        )
    return times, prefixes


@njit(nogil=True, cache=True)
def _magnus_build(h0, x, y, frequency, tolerance):
    period = 1 / abs(frequency)
    # Keep both grids away from a full spectral turn, where sampled carrier
    # motion can alias. The retained grid stays below half a spectral turn.
    width = (
        np.max(h0.real) - np.min(h0.real) + 2 * (np.linalg.norm(x) + np.linalg.norm(y))
    )
    count = max(8, 8 * int(np.ceil(1.5 * period * width / 8)))
    _, coarse = _magnus_prefixes(h0, x, y, frequency, count)
    return _magnus_refine(_Hamiltonian(h0, x, y, frequency), tolerance, count, coarse)


@njit(nogil=True, cache=True)
def _magnus_refine(context, tolerance, count, coarse):
    h0, x, y, frequency = context
    # A rejected continuation has already built the independent coarse grid.
    for _ in range(12):
        finer = 8 * int(np.ceil(1.5 * count / 8))
        times, fine = _magnus_prefixes(h0, x, y, frequency, finer)
        error = 0.0
        # Compare micromotion at eight phase intervals, not just U(T), where
        # errors can cancel. Both grids include these endpoints exactly.
        for j in range(9):
            error = max(
                error, np.linalg.norm(coarse[j * count // 8] - fine[j * finer // 8])
            )
        if error <= tolerance:
            return times, fine
        count = finer
        coarse = fine
    raise RuntimeError("Floquet Magnus preparation failed to converge")


@njit(nogil=True, cache=True)
def _dense(times, coefficients, time):
    step = min(np.searchsorted(times, time, side="right") - 1, len(times) - 2)
    step = max(0, step)
    scaled = (time - times[step]) / (times[step + 1] - times[step])
    result = np.zeros_like(coefficients[step, 0])
    for index in range(7):
        result += coefficients[step, 7 - index]
        if index % 2 == 0:
            result *= scaled
        else:
            result *= 1 - scaled
    return coefficients[step, 0] + result


@njit(nogil=True, cache=True)
def _dissipate(rho, population, coherence):
    diagonal = np.diag(rho).copy()
    result = rho * coherence
    updated = population @ diagonal
    for index in range(len(updated)):
        result[index, index] = updated[index]
    return result


@njit(nogil=True, cache=True)
def _advance(rho, unitary, population, coherence, count):
    adjoint = unitary.conj().T.copy()
    for _ in range(count):
        rho = _dissipate(rho, population, coherence)
        rho = unitary @ rho @ adjoint
        rho = _dissipate(rho, population, coherence)
    return rho


@njit(nogil=True, cache=True)
def _advance_segments(rho, unitaries, population, coherence, count):
    for _ in range(count):
        for unitary in unitaries:
            rho = _advance(rho, unitary, population, coherence, 1)
    return rho


@njit(nogil=True, cache=True)
def _dense_fundamental(times, coefficients, unitary, time):
    period = times[-1]
    local = time % period
    if local == 0:
        return unitary if time >= period else np.eye(len(unitary), dtype=np.complex128)
    value = _unitary(_dense(times, coefficients, local))
    if time >= period:
        value = value @ unitary
    return value


@njit(nogil=True, cache=True)
def _dense_row(entry, origin, duration, rho, tail_bath):
    # Keep endpoint playback, polar correction and all density steps in one
    # nogil call. Tiny dispatcher calls otherwise contend on the worker GIL.
    start = _dense_fundamental(entry.times, entry.coefficients, entry.unitary, origin)
    unitary = start @ entry.unitary @ start.conj().T
    period = entry.times[-1]
    count = int(duration // period)
    tail = duration - count * period
    result = _advance(rho, unitary, entry.population_half, entry.coherence_half, count)
    if tail > 0:
        end = _dense_fundamental(
            entry.times, entry.coefficients, entry.unitary, origin + tail
        )
        result = _advance(result, end @ start.conj().T, tail_bath[0], tail_bath[1], 1)
    return result


def _fundamental(entry: _Period, time: float) -> NDArray[np.complex128]:
    if entry.magnus is None:
        return _dense_fundamental(entry.times, entry.coefficients, entry.unitary, time)
    period = float(entry.times[-1])
    local = time % period
    if local == 0:
        # Exact boundaries need neither dense playback nor polar correction.
        return (
            entry.unitary
            if time >= period
            else np.eye(len(entry.unitary), dtype=complex)
        )
    index = min(
        int(np.searchsorted(entry.times, local, side="right")) - 1,
        len(entry.times) - 2,
    )
    h0, x, y, frequency = entry.magnus
    value = (
        _magnus_step(
            h0, x, y, frequency, entry.times[index], local - entry.times[index]
        )
        @ entry.coefficients[index]
    )
    value = _unitary(value)
    if time >= period:
        value = value @ entry.unitary
    return value


def _subdivided_row(
    entry: _Period,
    origin: float,
    duration: float,
    rho: NDArray[np.complex128],
    subdivisions: int,
) -> NDArray[np.complex128]:
    period = float(entry.times[-1])
    step = period / subdivisions
    boundaries = [
        _fundamental(entry, origin + j * step) for j in range(subdivisions + 1)
    ]
    unitaries = np.array(
        [
            end @ start.conj().T
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        ]
    )
    population = expm(entry.generator * step / 2).astype(complex)
    coherence = np.exp(-entry.decay * step / 2)
    count = int(duration // period)
    result = _advance_segments(rho, unitaries, population, coherence, count)
    tail = duration - count * period
    whole = min(int(tail // step), subdivisions - 1)
    result = _advance_segments(result, unitaries[:whole], population, coherence, 1)
    residual = tail - whole * step
    if residual > 0:
        unitary = _fundamental(entry, origin + tail) @ boundaries[whole].conj().T
        result = _advance(
            result,
            unitary,
            expm(entry.generator * residual / 2).astype(complex),
            np.exp(-entry.decay * residual / 2),
            1,
        )
    return result


def _row(
    entry: _Period,
    frequency: float,
    theta: float,
    duration: float,
    rho: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    if duration == 0:
        return rho
    period = float(entry.times[-1])
    origin = (theta / (2 * np.pi * frequency)) % period
    subdivisions = _subdivisions(duration * entry.split_rate)
    if subdivisions > 1:
        return _subdivided_row(entry, origin, duration, rho, subdivisions)
    if entry.magnus is None:
        tail = duration - int(duration // period) * period
        tail_bath = (entry.population_half, entry.coherence_half)
        if tail > 0:
            tail_bath = (
                expm(entry.generator * tail / 2).astype(complex),
                np.exp(-entry.decay * tail / 2),
            )
        dense = _DensePeriod(
            entry.times,
            entry.coefficients,
            entry.unitary,
            entry.population_half,
            entry.coherence_half,
        )
        return _dense_row(dense, origin, duration, rho, tail_bath)
    start = _fundamental(entry, origin)
    unitary = start @ entry.unitary @ start.conj().T
    count = int(duration // period)
    tail = duration - count * period
    result = _advance(rho, unitary, entry.population_half, entry.coherence_half, count)
    if tail > 0:
        end = origin + tail
        tail_unitary = _fundamental(entry, end)
        tail_unitary = tail_unitary @ start.conj().T
        result = _advance(
            result,
            tail_unitary,
            expm(entry.generator * tail / 2).astype(complex),
            np.exp(-entry.decay * tail / 2),
            1,
        )
    return result


class FloquetPeriodKernel:
    """One constant tone, independent of input density and absolute start.

    Keys contain only level, port, signed frequency (GHz) and signed gain.
    Preparation builds one Hilbert-space U(T), retaining U(T) and finite
    endpoint data (CF4 prefixes or degree-seven DOP853 coefficients). No Liouville
    channel is built. Theta translates the time origin by theta/(2*pi*f) mod T;
    each subperiod and the separately evaluated fractional tail apply D/2, U, D/2.
    Known and unknown densities, including coherent inputs, use the same channel.

    The density-free guard requires |f| >= 0.03 GHz and selects the smallest
    m in 1..64 such that duration*T**2*Gamma*V**2/m**2 <= 1.0. Here T=1/|f|
    is in ns, V=2*pi*|gain|*(||X||_2+||Y||_2) in rad/ns, and Gamma is the
    larger of the population generator norm and coherence decay in 1/ns.
    This retains the calibrated splitting allowance, not a rigorous total-error
    bound or segment-budget guarantee. Each subperiod uses unitary Hilbert
    endpoints from the same Hilbert trajectory. The tail uses the same step ceiling.
    The subdivision count depends only on model and row, never density; it is
    not part of the preparation key. Rows outside this domain still try T6.

    Below 1 GHz, preparation uses fourth-order commutator-free Magnus (CF4)
    with scaling-and-squaring [7/7] Pade exponentials at double precision,
    retaining unitary prefixes. The matrix one-norm selects scaling at the
    0.9504178996162932 backward-error radius. Centering each exponential removes
    its common energy during evaluation and restores its scalar phase afterward.
    A fractional endpoint uses a CF4 step from its preceding prefix, not
    interpolation. Uniform grids start below 2/3 of a spectral turn and compare prefix
    trajectories with a 1.5x finer grid at eight phase intervals. The maximum
    Frobenius defect must be <= min(1e-4, budget*0.1); the finer trajectory is
    retained. Its steps are below half a spectral turn.

    Within an Engine evolve, fixed-carrier groups with multiple gains may
    replace the fine grid with gain continuation. The complete table's sorted
    unique gains form up to twelve interleaved trajectories, with at least
    sixteen gains per trajectory before adding another. Each trajectory splits
    into fixed blocks of at most 32 gains. This plan never reads worker count,
    chunk size or execution order. Under one block lock, any caller builds the
    fixed predecessors up to its requested key. The last four built coordinates
    support prediction and memoize unclaimed results. Handoff removes a key's
    memo; older unclaimed keys are evicted rather than retaining unused scan
    prefixes. Completing a block releases its history. A key no longer memoized
    replays the same fixed predecessors. A caller holds only one block lock.
    Tables are bitwise invariant to workers, chunks and request order for the
    same table, source and budget. Workspace bytes charge history and memo.

    History requires the identical model object, port, signed carrier and
    budget. Two tables give a linear predictor; three/four give quadratic/cubic
    Lagrange extrapolation of interaction-picture uniform prefixes. Because H0
    and the grid are identical, the analytic H0 transform and inverse cancel in
    the weighted sum, avoiding redundant phase exponentials. Repeated gains,
    unequal retained grid counts, a last gain-gap ratio above two, or a retained
    count below this key's first fine
    grid disable the predictor. Carrier continuation is never attempted.
    The candidate must pass the same independent coarse-grid comparison at
    eight phase intervals, including U(T), with min(1e-4, budget*0.1). Failure
    reuses that coarse grid in the original twelve-refinement path. Every
    candidate keeps the new key's H0, X, Y, frequency, bath and split rate.
    Standalone calls without a scope keep independent coarse/fine preparation.

    This is calibrated trajectory comparison, not a rigorous bound. Eight-phase
    defects may miss between-phase error. Continued prefixes are not products
    of unitary exponentials; their closeness to unitary for one Newton polar
    step is unproven. Gain smoothness and well-conditioned Lagrange weights are
    also unproven for the planned gain gaps. Model immutability is assumed.
    No history crosses evolve calls or enters the LRU.
    Duration remains outside the payload and cache key.

    Higher carriers use DOP853 tolerance min(1e-10, budget*1e-4), max_step=T/4.
    Both builders run in single-threaded nogil code. The high-carrier payload
    keeps the dense polynomial's native nested coefficients, without resampling
    or a per-key basis conversion.
    Only accepted steps are retained; no solver object is kept. One Newton polar
    step restores unitarity of U(T) and fundamental solutions before phase
    translation and tail composition. The tight integration controls and CF4's
    Hermitian exponentials keep endpoints near unitary, so the correction squares
    their norm defect without an SVD. Exact period boundaries reuse identity or
    U(T). Unsubdivided high-carrier rows compose endpoints and step density in
    one nogil call. Density is never normalized. Prototype high-carrier S3/S5 population
    errors were 1.1e-7/2.3e-6.

    Invalid budgets/representatives/payloads raise ValueError. Integration or
    nonfinite numeric output raises RuntimeError, never retries or reroutes.
    All prepared arrays are immutable; nbytes charges every retained array.
    """

    declaration = KernelDeclaration(
        identity=AlgorithmIdentity("floquet_period", "cf4-gain-continuation-v4"),
        eligibility=Eligibility(1, 1, "constant"),
        guard=GuardDeclaration(),
        accuracy="guarded-approximation",
        reuse=ReuseDeclaration("floquet_period/cf4-gain-continuation-v4/serial"),
        fallback="legacy-order",
    )

    name = "floquet_period"
    guard_builds = False

    def __init__(self, workspace: _GainWorkspace | None = None) -> None:
        """Use an evolve-local scope, or independent builds for standalone calls."""
        self._workspace = workspace

    def evolve_scope(
        self, table: SegmentTable, source: ModelSource
    ) -> tuple["FloquetPeriodKernel", _GainWorkspace]:
        """Plan deterministic gain blocks in a fresh byte-counted workspace.

        table is the complete segmented scan; source is this evolve's model
        owner. Numeric tables are independent of worker count, chunking and
        request order for this table/source/budget. No density is read.
        Engine shares the scoped kernel with prefix
        and timeline workers and drops both returned objects when evolve ends.
        Standalone prepare without this scope retains independent CF4 builds.
        """
        del source
        workspace = _GainWorkspace(table)
        # Preserve injected subclass guards/apply hooks without mutating the
        # registry instance shared by concurrent evolves.
        scoped = copy(self)
        scoped._workspace = workspace
        return scoped, workspace

    def guard(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> NDArray[np.bool_]:
        """Return row-local constant-tone, carrier and splitting-estimate validity."""
        _check_budget(budget)
        accepted = np.zeros(len(rows), dtype=np.bool_)
        for index, row in enumerate(rows):
            if table.tones[row] != 1:
                continue
            envelope = table.envelopes[int(table.tone_envelope[row, 0])]
            frequency = abs(float(table.tone_frequency_ghz[row, 0]))
            if envelope.kind != "constant" or frequency < _MIN_FREQUENCY_GHZ:
                continue
            model = source.model(tuple(int(v) for v in table.level[row]))
            estimate = table.duration_ns[row] * _split_rate(
                model,
                int(table.tone_port[row, 0]),
                frequency,
                float(table.tone_amplitude[row, 0]),
            )
            accepted[index] = estimate <= _MAX_SPLIT_ESTIMATE * _MAX_SUBDIVISIONS**2
        return accepted

    def prepare_keys(
        self, table: SegmentTable, rows: NDArray[np.int64]
    ) -> NDArray[np.void]:
        """Encode level/port/frequency/gain, excluding duration, start and theta."""
        keys = np.empty(
            len(rows),
            dtype=[
                ("level", "<i8", table.level.shape[1]),
                ("port", "<i8"),
                ("frequency", "<f8"),
                ("gain", "<f8"),
            ],
        )
        keys["level"] = table.level[rows]
        if len(rows):
            if np.any(table.tones[rows] != 1):
                raise ValueError("Floquet period keys require one tone")
            keys["port"] = table.tone_port[rows, 0]
            keys["frequency"] = table.tone_frequency_ghz[rows, 0]
            keys["gain"] = table.tone_amplitude[rows, 0]
        return keys

    def cost(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.float64]:
        """Estimate density steps, proportional to duration times carrier frequency."""
        return 1 + table.duration_ns[rows] * np.abs(table.tone_frequency_ghz[rows, 0])

    def prepare(
        self,
        keys: NDArray[np.void],
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> Prepared:
        """Build immutable one-period Hilbert data per aligned representative key."""
        _check_budget(budget)
        if len(keys) != len(rows) or not np.array_equal(
            keys, self.prepare_keys(table, rows)
        ):
            raise ValueError("Floquet period representatives must match keys")
        if not self.guard(table, rows, source, budget).all():
            raise ValueError("Floquet period representatives must pass guard")
        return _Periods(
            tuple(
                (_build if self._workspace is None else self._workspace.build)(
                    source.model(tuple(int(v) for v in key["level"])),
                    int(key["port"]),
                    float(key["frequency"]),
                    float(key["gain"]),
                    budget,
                )
                for key in keys
            )
        )

    def guard_prepared(
        self,
        table: SegmentTable,
        rows: NDArray[np.int64],
        source: ModelSource,
        budget: float,
    ) -> None:
        """Decline handoff: the scalar guard builds no propagator."""
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
        """Decline input binding: this channel applies to arbitrary densities."""
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
        """Advance aligned states in place; return all False without a posterior check.

        Uses only table duration, signed frequency and effective start theta.
        No refinement, normalization, retry or dispatch occurs here. Prepared
        data remain read-only across reordered/split calls and worker threads.
        """
        _check_budget(budget)
        if not isinstance(prepared, _Periods):
            raise ValueError("Floquet period requires its own prepared payload")
        for index, row in enumerate(rows):
            states[index] = _row(
                prepared.entries[int(key_index[index])],
                float(table.tone_frequency_ghz[row, 0]),
                float(table.tone_theta_rad[row, 0]),
                float(table.duration_ns[row]),
                states[index],
            )
        if not np.isfinite(states).all():
            raise RuntimeError("Floquet period apply produced nonfinite density")
        return np.zeros(len(rows), dtype=np.bool_)
