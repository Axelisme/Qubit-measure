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
_MAX_SUBDIVISIONS = 4


@dataclass(frozen=True)
class _Period:
    # Per-key arrays, including degree-seven dense U on each accepted ODE step.
    times: NDArray[np.float64]
    coefficients: NDArray[np.complex128]
    unitary: NDArray[np.complex128]
    generator: NDArray[np.float64]
    decay: NDArray[np.float64]
    population_half: NDArray[np.complex128]
    coherence_half: NDArray[np.float64]
    split_rate: float

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
        )


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
    # Polar projection removes integrator norm drift before thousands of steps.
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
    return _Period(*arrays, _split_rate(model, port, frequency, gain))


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
    value = _unitary(_dense(entry.times, entry.coefficients, time % period))
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
    start = _unitary(_dense(entry.times, entry.coefficients, origin))
    unitary = start @ entry.unitary @ start.conj().T
    count = int(duration // period)
    tail = duration - count * period
    result = _advance(rho, unitary, entry.population_half, entry.coherence_half, count)
    if tail > 0:
        end = origin + tail
        tail_unitary = _unitary(_dense(entry.times, entry.coefficients, end % period))
        if end >= period:
            tail_unitary = tail_unitary @ entry.unitary
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
    Preparation integrates one Hilbert-space U(T), retaining U(T) and bounded
    degree-seven dense coefficients on its finite accepted steps. No Liouville
    channel is built. Theta translates the time origin by theta/(2*pi*f) mod T;
    each subperiod and the separately evaluated fractional tail apply D/2, U, D/2.
    Known and unknown densities, including coherent inputs, use the same channel.

    The density-free guard requires |f| >= 0.03 GHz and selects the smallest
    m in 1..4 such that duration*T**2*Gamma*V**2/m**2 <= 1.0. Here T=1/|f|
    is in ns, V=2*pi*|gain|*(||X||_2+||Y||_2) in rad/ns, and Gamma is the
    larger of the population generator norm and coherence decay in 1/ns.
    This retains the calibrated splitting allowance, not a rigorous total-error
    bound or segment-budget guarantee. Each subperiod uses exact dense Hilbert
    endpoints from the same integration. The tail uses the same step ceiling.
    The subdivision count depends only on model and row, never density; it is
    not part of the preparation key. Rows outside this domain still try T6.

    Integration uses tolerance min(1e-10, budget*1e-4) and max_step=T/4.
    DOP853 stages, error control and degree-seven dense construction run in
    single-threaded nogil code. The payload keeps the dense polynomial's native
    nested coefficients, without resampling or a per-key basis conversion.
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
