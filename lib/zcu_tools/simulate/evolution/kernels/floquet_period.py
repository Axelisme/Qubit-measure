"""Constant-tone Hilbert propagators with compact secular Strang stepping."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy.linalg import expm

from ..kernel import Prepared, SegmentTable
from ..secular import ModelSource, SecularModel
from ._dop853 import Tolerances, integrate_dense

_MIN_FREQUENCY_GHZ = 0.03
_MAX_SPLIT_ESTIMATE = 1.0
_MAX_SUBDIVISIONS = 64
_MAGNUS_FREQUENCY_GHZ = 1.0


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


def _unitary(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the polar unitary of an integrated fundamental solution."""
    left, _, right = np.linalg.svd(matrix)
    return left @ right


def _build(
    model: SecularModel, port: int, frequency: float, gain: float, budget: float
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
        times, coefficients = _magnus_build(h0, x, y, frequency, tolerance)
        final = coefficients[-1]
        magnus = (h0, x, y, frequency)
        for array in (h0, x, y):
            array.setflags(write=False)
    else:
        tolerance = min(1e-10, budget * 1e-4)
        times, coefficients, final = integrate_dense(
            _rhs,
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
    energies, vectors = np.linalg.eigh(h)
    return (vectors * np.exp(-2j * np.pi * step * energies)) @ vectors.conj().T


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


def _fundamental(entry: _Period, time: float) -> NDArray[np.complex128]:
    period = float(entry.times[-1])
    local = time % period
    if entry.magnus is None:
        value = _unitary(_dense(entry.times, entry.coefficients, local))
    else:
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
    with exact Hermitian exponentials, retaining unitary prefixes. A fractional
    endpoint uses a CF4 step from its preceding prefix, not interpolation.
    Uniform grids start below 2/3 of a spectral turn and compare prefix
    trajectories with a 1.5x finer grid at eight phase intervals. The maximum
    operator defect must be <= min(1e-4, budget*0.1); the finer trajectory is
    retained. Its steps are below half a spectral turn. This is a calibrated
    numerical estimate, not a rigorous bound. It depends only on model and key, and keeps
    duration out of the payload and cache key.

    Higher carriers use DOP853 tolerance min(1e-10, budget*1e-4), max_step=T/4.
    Both builders run in single-threaded nogil code. The high-carrier payload
    keeps the dense polynomial's native nested coefficients, without resampling
    or a per-key basis conversion.
    Only accepted steps are retained; no solver object is kept. Polar projection of
    U(T) and dense fundamental solutions restores unitarity before phase
    translation and tail composition; density is never normalized. Measured
    coherent-input trace drift fell from 1.34e-10 to 1.68e-12. Prototype
    high-carrier S3/S5 population errors were 1.1e-7/2.3e-6.

    Invalid budgets/representatives/payloads raise ValueError. Integration or
    nonfinite numeric output raises RuntimeError, never retries or reroutes.
    All prepared arrays are immutable; nbytes charges every retained array.
    """

    name = "floquet_period"

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
                _build(
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
